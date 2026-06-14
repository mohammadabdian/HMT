import torch
from collections import defaultdict
from tqdm import tqdm
import time
from pathlib import Path
from datetime import datetime
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt", quiet=True)


class FastEvaluator:
    def __init__(self, model, tokenizer, device, eval_size=300, save_dir="evaluation_results"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.eval_size = eval_size
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.smoothing = SmoothingFunction().method1

    def get_subset(self, dataloader):
        subset = []
        collected = 0
        for batch in dataloader:
            features, captions, img_ids = batch
            batch_size = features.size(0)
            if collected + batch_size > self.eval_size:
                take = self.eval_size - collected
                if take > 0:
                    subset.append((features[:take], captions[:take], img_ids[:take]))
                break
            subset.append(batch)
            collected += batch_size
        return subset

    @torch.no_grad()
    def evaluate(self, test_loader, max_len=48, beam_size=3):
        self.model.eval()
        subset = self.get_subset(test_loader)

        print(f"\nEvaluating on {self.eval_size} images (beam={beam_size})...")

        start_time = time.time()
        refs_by_image = defaultdict(list)
        hyps_by_image = {}

        for features, captions, img_ids in tqdm(subset, desc="Generating"):
            features = features.to(self.device).float()
            generated_ids = self.generate_batch(features, max_len, beam_size)

            for i in range(features.size(0)):
                img_id = img_ids[i].item()
                ref_text = self.tokenizer.decode(captions[i], skip_special_tokens=True)
                hyp_text = self.tokenizer.decode(generated_ids[i], skip_special_tokens=True)

                refs_by_image[img_id].append(ref_text)
                hyps_by_image[img_id] = hyp_text

        metrics = self.compute_metrics_with_multiple_refs(refs_by_image, hyps_by_image)

        elapsed = time.time() - start_time
        self.print_results(metrics, elapsed)
        self.save_report(metrics, elapsed, beam_size, max_len)

        return metrics

    @torch.no_grad()
    def generate_batch(self, features, max_len=48, beam_size=3):
        """Beam Search"""
        batch_size = features.size(0)
        device = self.device
        bos_id = self.tokenizer.bos_token_id
        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id

        if beam_size == 1:
            return self.greedy_generate(features, max_len, bos_id, eos_id, pad_id)

        # Initialize
        sequences = torch.full((batch_size, beam_size, 1), bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(batch_size, beam_size, device=device)
        finished = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=device)

        for step in range(max_len - 1):
            if finished.all():
                break

            # Flatten for batch processing
            flat_seq = sequences.view(-1, sequences.size(-1))          # (B*beam, seq_len)
            flat_features = features.repeat_interleave(beam_size, dim=0)

            logits = self.model(flat_seq, flat_features)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            topk_log_probs, topk_ids = torch.topk(log_probs, beam_size, dim=-1)

            new_sequences = []
            new_scores = []

            for b in range(batch_size):
                for k in range(beam_size):
                    if finished[b, k]:
                        # Keep finished beam as is
                        new_sequences.append(sequences[b, k].clone())
                        new_scores.append(scores[b, k])
                        continue

                    idx = b * beam_size + k
                    # Choose best next token for this beam
                    best_k = topk_log_probs[idx].argmax()
                    next_token = topk_ids[idx, best_k]
                    next_score = topk_log_probs[idx, best_k]

                    new_seq = torch.cat([sequences[b, k], next_token.unsqueeze(0)])
                    new_sequences.append(new_seq)
                    new_scores.append(scores[b, k] + next_score)

                    if next_token == eos_id:
                        finished[b, k] = True

            # Stack with padding to handle different lengths
            max_current_len = max(seq.size(0) for seq in new_sequences)
            padded_seqs = []
            for seq in new_sequences:
                if seq.size(0) < max_current_len:
                    pad = torch.full((max_current_len - seq.size(0),), pad_id, 
                                   dtype=torch.long, device=device)
                    seq = torch.cat([seq, pad])
                padded_seqs.append(seq)

            sequences = torch.stack(padded_seqs).view(batch_size, beam_size, -1)
            scores = torch.stack(new_scores).view(batch_size, beam_size)

        # Select best beam per image
        best_idx = scores.argmax(dim=1)
        final_tokens = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)

        for i in range(batch_size):
            best_seq = sequences[i, best_idx[i]]
            valid_len = (best_seq != pad_id).sum().item() if (best_seq == pad_id).any() else len(best_seq)
            final_tokens[i, :valid_len] = best_seq[:valid_len]

        return final_tokens

    @torch.no_grad()
    def greedy_generate(self, features, max_len, bos_id, eos_id, pad_id):
        batch_size = features.size(0)
        sequences = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=self.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(max_len - 1):
            if finished.all():
                break
            logits = self.model(sequences, features)
            next_tokens = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            sequences = torch.cat([sequences, next_tokens], dim=1)
            finished = finished | (next_tokens.squeeze(-1) == eos_id)

        padded = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=self.device)
        for i in range(batch_size):
            length = min(sequences[i].size(0), max_len)
            padded[i, :length] = sequences[i][:length]
        return padded
    
    def compute_metrics_with_multiple_refs(self, refs_by_image, hyps_by_image):
        print("Computing metrics with multiple references per image...")
        
        all_meteor_scores = []
        all_rouge_scores = []
        
        all_references = []
        all_hypotheses = []
        
        cider_refs = {}
        cider_hyps = {}
        
        for idx, (img_id, refs) in enumerate(refs_by_image.items()):
            hypothesis = hyps_by_image[img_id]
            
            refs_tokenized = [ref.split() for ref in refs]
            hyp_tokenized = hypothesis.split()
            
            all_references.append(refs_tokenized)
            all_hypotheses.append(hyp_tokenized)
            
            # METEOR: best score over all references
            best_meteor = max(meteor_score([ref], hyp_tokenized) for ref in refs_tokenized)
            all_meteor_scores.append(best_meteor)
            
            # ROUGE-L: best score over all references
            best_rouge = 0
            for ref in refs:
                rouge_score = self.scorer.score(ref, hypothesis)['rougeL'].fmeasure
                best_rouge = max(best_rouge, rouge_score)
            all_rouge_scores.append(best_rouge)
            
            # CIDEr و SPICE
            cider_refs[idx] = refs
            cider_hyps[idx] = [hypothesis]
        
        bleu_scores = {}
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0/n]*n) + tuple([0.0]*(4-n))
            bleu_scores[f'BLEU-{n}'] = self.corpus_bleu_with_multiple_refs(all_references, all_hypotheses, weights)
        
        meteor = sum(all_meteor_scores) / len(all_meteor_scores) if all_meteor_scores else 0
        rouge = sum(all_rouge_scores) / len(all_rouge_scores) if all_rouge_scores else 0
        
        # CIDEr
        cider_score, _ = Cider().compute_score(cider_refs, cider_hyps)
        
        # SPICE
        try:
            spice_score, _ = Spice().compute_score(cider_refs, cider_hyps)
        except Exception:
            spice_score = 0.0
        
        metrics = {
            'BLEU-1': bleu_scores['BLEU-1'],
            'BLEU-2': bleu_scores['BLEU-2'],
            'BLEU-3': bleu_scores['BLEU-3'],
            'BLEU-4': bleu_scores['BLEU-4'],
            'METEOR': meteor,
            'ROUGE_L': rouge,
            'CIDEr': float(cider_score),
            'SPICE': float(spice_score),
            'num_images': len(refs_by_image),
            'avg_references_per_image': sum(len(refs) for refs in refs_by_image.values()) / len(refs_by_image)
        }
        
        print(f"  Average references per image: {metrics['avg_references_per_image']:.1f}")
        
        return metrics
    
    def corpus_bleu_with_multiple_refs(self, references_list, hypotheses, weights):
        from nltk.translate.bleu_score import corpus_bleu
        return corpus_bleu(references_list, hypotheses, weights)
    
    def print_results(self, metrics, elapsed_time):
        print("\n" + "="*50)
        print(f"RESULTS - {metrics['num_images']} images - {elapsed_time:.1f}s")
        print(f"Avg references per image: {metrics['avg_references_per_image']:.1f}")
        print("="*50)
        print(f"BLEU-1: {metrics['BLEU-1']:.4f}")
        print(f"BLEU-2: {metrics['BLEU-2']:.4f}")
        print(f"BLEU-3: {metrics['BLEU-3']:.4f}")
        print(f"BLEU-4: {metrics['BLEU-4']:.4f}")
        print(f"METEOR: {metrics['METEOR']:.4f}")
        print(f"ROUGE-L: {metrics['ROUGE_L']:.4f}")
        print(f"CIDEr: {metrics['CIDEr']:.4f}")
        print(f"SPICE: {metrics['SPICE']:.4f}")
        print("="*50)
    
    def save_report(self, metrics, detailed_results, elapsed_time, beam_size, max_len):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        report_file = self.save_dir / f"eval_{timestamp}.txt"
        json_file = self.save_dir / f"eval_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("="*50 + "\n")
            f.write("EVALUATION REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Images: {metrics['num_images']}\n")
            f.write(f"Avg references per image: {metrics['avg_references_per_image']:.1f}\n")
            f.write(f"Beam: {beam_size}\n")
            f.write(f"Time: {elapsed_time:.1f}s\n\n")
            f.write("-"*50 + "\n")
            f.write("METRICS\n")
            f.write("-"*50 + "\n")
            for metric, score in metrics.items():
                if not isinstance(score, (int, float)):
                    continue
                f.write(f"{metric:<15} {score:.4f}\n")
            f.write("="*50 + "\n")
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nSaved: {report_file}")
    
    def print_samples(self, detailed_results, num_samples=5):
        print("\n" + "="*50)
        print("SAMPLE OUTPUTS")
        print("="*50)
        for i, sample in enumerate(detailed_results[:num_samples]):
            print(f"\n[{i+1}] Image ID: {sample['image_id']}")
            print(f"Ref (1/{sample['num_references']}): {sample['reference']}")
            print(f"Gen: {sample['hypothesis']}")
        print("="*50)


def evaluate_bleu(model, test_loader, tokenizer, device, num_samples=300, beam_size=3, save_dir="evaluation_results"):
    evaluator = FastEvaluator(model, tokenizer, device, eval_size=num_samples, save_dir=save_dir)
    metrics, detailed_results = evaluator.evaluate(test_loader, beam_size=beam_size)
    evaluator.print_samples(detailed_results, num_samples=5)
    return metrics


def evaluate_bleu_with_details(model, test_loader, tokenizer, device, num_samples=300, beam_size=3, save_dir="evaluation_results"):
    evaluator = FastEvaluator(model, tokenizer, device, eval_size=num_samples, save_dir=save_dir)
    metrics, detailed_results = evaluator.evaluate(test_loader, beam_size=beam_size)
    evaluator.print_samples(detailed_results, num_samples=5)
    return metrics, detailed_results


def evaluate_fast(model, test_loader, tokenizer, device, num_samples=300, beam_size=3, save_dir="evaluation_results"):
    return evaluate_bleu(model, test_loader, tokenizer, device, num_samples, beam_size, save_dir)
