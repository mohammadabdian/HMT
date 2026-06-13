import torch
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from tqdm import tqdm
from collections import defaultdict
from generate import generate_caption
import nltk

nltk.data.path.append('/home/ubuntu/project/nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download("punkt")


def compute_metrics(
    model,
    test_loader,
    tokenizer,
    device
):
    model.eval()
    
    refs_by_image = defaultdict(list)
    hyps_by_image = {}
    
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    
    with torch.no_grad():
        for image_features, captions, image_ids in tqdm(test_loader, desc="Computing metrics"):
            image_features = image_features.to(device)
            captions = captions.to(device)
            
            generated = generate_caption(model, image_features, tokenizer)
            
            for i in range(captions.size(0)):
                img_id = image_ids[i].item()
                
                ref_tokens = captions[i].tolist()
                hyp_tokens = generated[i].tolist()
                
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                hyp_text = tokenizer.decode(hyp_tokens, skip_special_tokens=True)
                
                refs_by_image[img_id].append(ref_text)
                hyps_by_image[img_id] = hyp_text
    
    meteor_scores = []
    rouge_scores = []
    cider_refs = {}
    cider_hyps = {}
    spice_refs = {}
    spice_hyps = {}
    
    for idx, img_id in enumerate(refs_by_image.keys()):
        references = refs_by_image[img_id]
        hypothesis = hyps_by_image[img_id]
        
        best_meteor = max([meteor_score([r.split()], hypothesis.split()) for r in references])
        best_rouge = max([rouge.score(r, hypothesis)["rougeL"].fmeasure for r in references])
        
        meteor_scores.append(best_meteor)
        rouge_scores.append(best_rouge)
        
        cider_refs[idx] = references
        cider_hyps[idx] = [hypothesis]
        
        spice_refs[idx] = references
        spice_hyps[idx] = [hypothesis]
    
    cider = Cider()
    spice = Spice()
    
    cider_score, _ = cider.compute_score(cider_refs, cider_hyps)
    spice_score, _ = spice.compute_score(spice_refs, spice_hyps)
    
    results = {
        "METEOR": sum(meteor_scores) / len(meteor_scores),
        "ROUGE_L": sum(rouge_scores) / len(rouge_scores),
        "CIDEr": cider_score,
        "SPICE": spice_score
    }
    
    print(f"METEOR: {results['METEOR']:.4f}")
    print(f"ROUGE-L: {results['ROUGE_L']:.4f}")
    print(f"CIDEr: {results['CIDEr']:.4f}")
    print(f"SPICE: {results['SPICE']:.4f}")
    
    return results


def evaluate_bleu(
    model,
    test_loader,
    tokenizer,
    device
):
    model.eval()
    
    refs_by_image = defaultdict(list)
    hyps_by_image = {}
    
    with torch.no_grad():
        for image_features, captions, image_ids in tqdm(test_loader, desc="Evaluating BLEU"):
            image_features = image_features.to(device)
            captions = captions.to(device)
            
            generated = generate_caption(model, image_features, tokenizer)
            
            for i in range(captions.size(0)):
                img_id = image_ids[i].item()
                
                ref_tokens = captions[i].tolist()
                hyp_tokens = generated[i].tolist()
                
                ref_text = tokenizer.decode(ref_tokens, skip_special_tokens=True)
                hyp_text = tokenizer.decode(hyp_tokens, skip_special_tokens=True)
                
                refs_by_image[img_id].append(ref_text.split())
                hyps_by_image[img_id] = hyp_text.split()
    
    references = []
    hypotheses = []
    
    for img_id in refs_by_image.keys():
        references.append(refs_by_image[img_id])
        hypotheses.append(hyps_by_image[img_id])
    
    from nltk.translate.bleu_score import corpus_bleu
    
    bleu_1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu_4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    print(f"BLEU-1: {bleu_1:.4f}")
    print(f"BLEU-2: {bleu_2:.4f}")
    print(f"BLEU-3: {bleu_3:.4f}")
    print(f"BLEU-4: {bleu_4:.4f}")
    
    return {
        "BLEU-1": bleu_1,
        "BLEU-2": bleu_2,
        "BLEU-3": bleu_3,
        "BLEU-4": bleu_4
    }