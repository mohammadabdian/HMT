import torch
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class GenerationConfig:
    max_len: int = 48
    beam_size: int = 1
    alpha: float = 0.6
    temperature: float = 1.2
    top_p: float = 0.95
    use_sampling: bool = True
    repetition_penalty: float = 1.3


class CaptionGenerator:
    def __init__(self, model, tokenizer, config: Optional[GenerationConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GenerationConfig()
        self._device = next(model.parameters()).device

        self.bos_id = tokenizer.bos_token_id
        self.pad_id = tokenizer.pad_token_id

        self.eos_id = None
        for candidate in ["[EOS_NEW]", "[EOS]", "</s>", "<|endoftext|>"]:
            if candidate in tokenizer.get_vocab():
                self.eos_id = tokenizer.convert_tokens_to_ids(candidate)
                break
        if self.eos_id is None:
            self.eos_id = tokenizer.eos_token_id

        assert self.bos_id is not None, "tokenizer has no bos_token_id"
        assert self.eos_id is not None, "tokenizer has no eos_token_id"
        assert self.pad_id is not None, "tokenizer has no pad_token_id"

        self._special_ids = torch.tensor(
            [self.bos_id, self.pad_id, self.eos_id],
            dtype=torch.long, device=self._device
        )

        print(f"BOS={self.bos_id}, EOS={self.eos_id}, PAD={self.pad_id}")

    @property
    def device(self) -> torch.device:
        return self._device

    def _get_logits(self, sequences: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        output = self.model(sequences, features)
        if isinstance(output, torch.Tensor):
            return output[:, -1, :]
        if hasattr(output, "logits"):
            return output.logits[:, -1, :]
        raise ValueError(f"Unexpected model output type: {type(output)}")

    def _apply_temperature(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        if temperature == 1.0 or temperature == 0.0:
            return logits
        return logits / temperature

    def _top_p_filter(self, probs: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        remove = (cumsum - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        probs_out = torch.zeros_like(probs)
        probs_out.scatter_(-1, sorted_idx, sorted_probs)
        return probs_out

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, sequences: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        if penalty == 1.0:
            return logits

        vocab_size = logits.size(-1)

        valid = (sequences >= 0) & (sequences < vocab_size)
        is_special = torch.isin(sequences, self._special_ids)
        keep = valid & ~is_special

        token_mask = torch.zeros(
            logits.size(0), vocab_size, dtype=torch.bool, device=self.device
        )
        safe_seqs = sequences.masked_fill(~keep, 0)
        token_mask.scatter_(1, safe_seqs, keep)

        logits[token_mask] = logits[token_mask] / penalty
        return logits

    def _pad_to(self, sequences: torch.Tensor, max_len: int) -> torch.Tensor:
        batch_size, seq_len = sequences.shape
        if seq_len >= max_len:
            return sequences[:, :max_len]
        pad = torch.full(
            (batch_size, max_len - seq_len), self.pad_id,
            dtype=torch.long, device=self.device
        )
        return torch.cat([sequences, pad], dim=1)

    def _pad_list(self, seqs: List[torch.Tensor], max_len: int) -> torch.Tensor:
        out = torch.full(
            (len(seqs), max_len), self.pad_id, dtype=torch.long, device=self.device
        )
        for i, s in enumerate(seqs):
            length = min(s.size(0), max_len)
            out[i, :length] = s[:length]
        return out

    @torch.no_grad()
    def _greedy_or_sampling(
        self, features: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        batch_size = features.size(0)
        sequences = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=self.device
        )
        finished = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        for _ in range(config.max_len - 1):
            if finished.all():
                break

            logits = self._get_logits(sequences, features)

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, sequences, config.repetition_penalty
                )

            if config.use_sampling and config.temperature > 0.0:
                logits = self._apply_temperature(logits, config.temperature)
                probs = torch.softmax(logits, dim=-1)
                if config.top_p < 1.0:
                    probs = self._top_p_filter(probs, config.top_p)
                next_tokens = torch.multinomial(probs, 1)
            else:
                next_tokens = logits.argmax(dim=-1, keepdim=True)

            next_tokens = next_tokens.masked_fill(finished.unsqueeze(-1), self.pad_id)
            finished = finished | (next_tokens.squeeze(-1) == self.eos_id)
            sequences = torch.cat([sequences, next_tokens], dim=1)

        return self._pad_to(sequences, config.max_len)

    @torch.no_grad()
    def _beam_search(
        self, features: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        batch_size  = features.size(0)
        beam_size   = config.beam_size
        max_len     = config.max_len
        alpha       = config.alpha
        temperature = config.temperature

        init_seq    = torch.full(
            (batch_size, 1), self.bos_id, dtype=torch.long, device=self.device
        )
        init_logits = self._get_logits(init_seq, features)
        init_logits = self._apply_temperature(init_logits, temperature)
        init_lp     = F.log_softmax(init_logits, dim=-1)

        scores, first_tokens = torch.topk(init_lp, beam_size, dim=-1)

        sequences = torch.stack([
            torch.full(
                (batch_size, beam_size), self.bos_id, dtype=torch.long, device=self.device
            ),
            first_tokens,
        ], dim=-1)

        finished = (first_tokens == self.eos_id)

        expanded_features = features.repeat_interleave(beam_size, dim=0)

        for _ in range(max_len - 2):
            if finished.all():
                break

            seq_len  = sequences.size(-1)
            flat_seq = sequences.view(batch_size * beam_size, seq_len)

            logits = self._get_logits(flat_seq, expanded_features)
            logits = self._apply_temperature(logits, temperature)

            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits, flat_seq, config.repetition_penalty
                )

            log_probs = F.log_softmax(logits, dim=-1)
            vocab_size = log_probs.size(-1)
            log_probs = log_probs.view(batch_size, beam_size, vocab_size)

            fin_mask = finished.unsqueeze(-1).expand_as(log_probs)
            log_probs = log_probs.masked_fill(fin_mask, -1e9)
            log_probs[finished.unsqueeze(-1).expand(
                batch_size, beam_size, 1
            ).squeeze(-1), :] = -1e9
            log_probs[:, :, self.pad_id] = torch.where(
                finished, torch.zeros(1, device=self.device),
                log_probs[:, :, self.pad_id]
            )

            cand = scores.unsqueeze(-1) + log_probs

            topk_scores, topk_ids = torch.topk(
                cand.view(batch_size, beam_size * vocab_size), beam_size, dim=-1
            )

            beam_ids  = topk_ids // vocab_size
            token_ids = topk_ids %  vocab_size

            idx = beam_ids.unsqueeze(-1).expand(batch_size, beam_size, seq_len)
            sequences = torch.gather(sequences, 1, idx)
            sequences = torch.cat(
                [sequences, token_ids.unsqueeze(-1)], dim=-1
            )

            scores  = topk_scores

            parent_finished = torch.gather(finished, 1, beam_ids)
            finished = parent_finished | (token_ids == self.eos_id)

        eos_mask  = sequences == self.eos_id
        has_eos   = eos_mask.any(dim=-1)
        eos_pos   = eos_mask.float().argmax(dim=-1)
        seq_lens  = torch.where(has_eos, eos_pos + 1,
                                torch.tensor(max_len, device=self.device))
        seq_lens  = seq_lens.clamp(min=1)

        norm_scores = scores / seq_lens.float().pow(alpha)
        best_beam   = norm_scores.argmax(dim=-1)

        best_idx  = best_beam.view(batch_size, 1, 1).expand(
            batch_size, 1, sequences.size(-1)
        )
        best_seqs = sequences.gather(1, best_idx).squeeze(1)

        return self._pad_to(best_seqs, max_len)

    def generate_batch(
        self, features: torch.Tensor, config: Optional[GenerationConfig] = None
    ) -> torch.Tensor:
        if config is None:
            config = self.config
        features = features.to(self.device).float()
        if config.beam_size > 1 and not config.use_sampling:
            return self._beam_search(features, config)
        return self._greedy_or_sampling(features, config)

    def generate_single(
        self, feature: torch.Tensor, config: Optional[GenerationConfig] = None
    ) -> str:
        if feature.dim() == 1:
            feature = feature.unsqueeze(0)
        tokens = self.generate_batch(feature, config)
        return self.tokenizer.decode(tokens[0], skip_special_tokens=True)

    def generate_dataloader(
        self,
        dataloader,
        config: Optional[GenerationConfig] = None,
        return_references: bool = True,
        show_progress: bool = True,
    ) -> List[Dict]:
        if config is None:
            config = self.config

        self.model.eval()
        results = []
        iterator = tqdm(dataloader, desc="Generating") if show_progress else dataloader

        for batch in iterator:
            if isinstance(batch, (list, tuple)):
                features = batch[0].to(self.device).float()
                captions = batch[1] if len(batch) >= 2 else None
                img_ids  = batch[2] if len(batch) >= 3 else None
            else:
                features = batch.to(self.device).float()
                captions = None
                img_ids  = None

            generated_tokens = self.generate_batch(features, config)

            for i in range(features.size(0)):
                img_id = (
                    (img_ids[i].item() if torch.is_tensor(img_ids[i]) else img_ids[i])
                    if img_ids is not None else len(results)
                )
                gen_text = self.tokenizer.decode(
                    generated_tokens[i], skip_special_tokens=True
                )
                result = {"image_id": img_id, "generated": gen_text}

                if return_references and captions is not None:
                    cap = captions[i]
                    if isinstance(cap, torch.Tensor):
                        result["references"] = [
                            self.tokenizer.decode(cap, skip_special_tokens=True)
                        ]
                    elif isinstance(cap, (list, tuple)):
                        result["references"] = [
                            self.tokenizer.decode(c, skip_special_tokens=True)
                            if isinstance(c, torch.Tensor) else c
                            for c in cap
                        ]
                    else:
                        result["references"] = [cap]

                results.append(result)

        return results