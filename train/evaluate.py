import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from functools import partial
from pathlib import Path

import nltk
import torch
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from rouge_score import rouge_scorer

from generator import CaptionGenerator, GenerationConfig

for _res in ["tokenizers/punkt", "corpora/wordnet", "corpora/omw-1.4"]:
    try:
        nltk.data.find(_res)
    except LookupError:
        nltk.download(_res.split("/")[-1], quiet=True)


def _meteor_single(args):
    refs_tok, hyp_tok = args
    
    if not hyp_tok or not refs_tok:
        return 0.0
    
    refs_tok = [[token for token in ref if token and token.strip()] for ref in refs_tok]
    hyp_tok = [token for token in hyp_tok if token and token.strip()]
    
    if not hyp_tok or not any(refs_tok):
        return 0.0
    
    try:
        return meteor_score(refs_tok, hyp_tok)
    except Exception:
        return 0.0


def _rouge_single(rouge_inst, args):
    refs, hyp = args
    if not refs:
        return 0.0
    try:
        return max(rouge_inst.score(ref, hyp)["rougeL"].fmeasure for ref in refs)
    except Exception:
        return 0.0


class FastEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        eval_size: int = 300,
        save_dir: str = "evaluation_results",
        num_workers: int = 4,
    ):
        self.generator = CaptionGenerator(model, tokenizer)
        self.eval_size = eval_size
        self.rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.smoother = SmoothingFunction().method1

    def _collect_subset(self, dataloader):
        subset, collected = [], 0
        for batch in dataloader:
            features, captions, img_ids = batch
            remaining = self.eval_size - collected
            if remaining <= 0:
                break
            if features.size(0) > remaining:
                features = features[:remaining]
                captions = captions[:remaining] if hasattr(captions, "__getitem__") else captions
                img_ids = img_ids[:remaining] if hasattr(img_ids, "__getitem__") else img_ids
            subset.append((features, captions, img_ids))
            collected += features.size(0)
            if collected >= self.eval_size:
                break
        return subset

    def _compute_metrics(self, refs_by_id: dict, hyps_by_id: dict) -> dict:
        img_ids = list(refs_by_id.keys())
        all_refs = [refs_by_id[i] for i in img_ids]
        all_hyps = [hyps_by_id[i] for i in img_ids]

        all_refs_tok = [[r.split() for r in refs] for refs in all_refs]
        all_hyps_tok = [h.split() for h in all_hyps]

        rouge_fn = partial(_rouge_single, self.rouge)

        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            try:
                meteor_scores = list(pool.map(_meteor_single, zip(all_refs_tok, all_hyps_tok)))
            except Exception:
                meteor_scores = [0.0] * len(all_hyps_tok)
            
            rouge_scores = list(pool.map(rouge_fn, zip(all_refs, all_hyps)))

        metrics = {}
        
        for n in [1, 2, 3, 4]:
            weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
            try:
                metrics[f"BLEU-{n}"] = corpus_bleu(
                    all_refs_tok, all_hyps_tok,
                    weights=weights,
                    smoothing_function=self.smoother,
                )
            except Exception:
                metrics[f"BLEU-{n}"] = 0.0

        sent_bleu = []
        for refs_tok, hyp_tok in zip(all_refs_tok, all_hyps_tok):
            try:
                sent_bleu.append(
                    sentence_bleu(refs_tok, hyp_tok, smoothing_function=self.smoother)
                )
            except Exception:
                pass
        metrics["BLEU_sent"] = sum(sent_bleu) / len(sent_bleu) if sent_bleu else 0.0

        metrics["METEOR"] = sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0
        metrics["ROUGE_L"] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0.0

        cider_refs = {i: refs for i, refs in enumerate(all_refs)}
        cider_hyps = {i: [hyp] for i, hyp in enumerate(all_hyps)}
        try:
            cider_score, _ = Cider().compute_score(cider_refs, cider_hyps)
            metrics["CIDEr"] = float(cider_score)
        except Exception:
            metrics["CIDEr"] = 0.0

        try:
            spice_scorer = Spice()
            spice_score, _ = spice_scorer.compute_score(cider_refs, cider_hyps)
            metrics["SPICE"] = float(spice_score)
        except Exception:
            metrics["SPICE"] = 0.0

        metrics["num_images"] = len(refs_by_id)
        metrics["avg_refs_per_image"] = (
            sum(len(v) for v in refs_by_id.values()) / len(refs_by_id)
            if refs_by_id else 0.0
        )
        return metrics

    def evaluate(self, dataloader, config: GenerationConfig = None) -> dict:
        if config is None:
            config = GenerationConfig(max_len=48, beam_size=3, alpha=0.6)

        subset = self._collect_subset(dataloader)
        total = sum(f.size(0) for f, _, _ in subset)
        print(f"\nEvaluating on {total} images ...")

        start = time.time()
        results = self.generator.generate_dataloader(
            subset, config, return_references=True, show_progress=True
        )

        refs_by_id: dict = defaultdict(set)
        hyps_by_id: dict = {}
        for r in results:
            img_id = r["image_id"]
            for ref in r.get("references", []):
                refs_by_id[img_id].add(ref)
            hyps_by_id[img_id] = r["generated"]

        refs_by_id = {k: list(v) for k, v in refs_by_id.items()}

        metrics = self._compute_metrics(refs_by_id, hyps_by_id)
        elapsed = time.time() - start

        self._print(metrics, elapsed)
        self._save(metrics, elapsed, config)
        return metrics

    _KEYS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "BLEU_sent", "METEOR", "ROUGE_L", "CIDEr", "SPICE"]

    def _print(self, metrics: dict, elapsed: float):
        print("\n" + "=" * 50)
        print(f"RESULTS — {metrics['num_images']} images — {elapsed:.1f}s")
        print(f"Avg refs/image: {metrics['avg_refs_per_image']:.1f}")
        print("=" * 50)
        for key in self._KEYS:
            print(f"{key:<12} {metrics.get(key, 0):.4f}")
        print("=" * 50)

    def _save(self, metrics: dict, elapsed: float, config: GenerationConfig):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        txt_path = self.save_dir / f"eval_{ts}.txt"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Date: {datetime.now()}\n")
            f.write(f"Images: {metrics['num_images']}\n")
            f.write(f"Avg refs/image: {metrics['avg_refs_per_image']:.1f}\n")
            f.write(
                f"Beam: {config.beam_size} | Max len: {config.max_len} | Time: {elapsed:.1f}s\n\n"
            )
            for key in self._KEYS:
                f.write(f"{key:<12} {metrics.get(key, 0):.4f}\n")

        json_path = self.save_dir / f"eval_{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))},
                f, indent=2,
            )
        print(f"Saved: {txt_path}")