
import torch

def generate_caption(
    model,
    image_features,
    tokenizer,
    max_len=48, # بر اساس خروجی شما که طولش 48 بود
    beam_size=3
):
    model.eval()
    image_features = image_features.float()
    batch_size = image_features.size(0)
    device = image_features.device

    bos_token_id = 49406
    eos_token_id = 49408
    pad_token_id = 49407

    beams = [
        [
            {
                "tokens": torch.full((1, 1), bos_token_id, dtype=torch.long, device=device),
                "score": 0.0
            }
        ] for _ in range(batch_size)
    ]

    with torch.no_grad():
        for _ in range(max_len - 1):
            all_candidates = []
            for i in range(batch_size):
                for beam in beams[i]:
                    tokens = beam["tokens"]
                    
                    if tokens[0, -1].item() == eos_token_id:
                        all_candidates.append({**beam, "batch_idx": i})
                        continue

                    logits = model(tokens, image_features[i:i+1])
                    log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                    topk_probs, topk_tokens = torch.topk(log_probs, beam_size, dim=-1)

                    for k in range(beam_size):
                        next_token = topk_tokens[0, k].unsqueeze(0).unsqueeze(0)
                        candidate_tokens = torch.cat([tokens, next_token], dim=1)
                        candidate_score = beam["score"] + topk_probs[0, k].item()
                        all_candidates.append({
                            "tokens": candidate_tokens,
                            "score": candidate_score,
                            "batch_idx": i
                        })

            new_beams = []
            for i in range(batch_size):
                candidates_i = [c for c in all_candidates if c["batch_idx"] == i]
                candidates_i = sorted(candidates_i, key=lambda x: x["score"], reverse=True)
                new_beams.append(candidates_i[:beam_size])
            beams = new_beams

            finished = True
            for i in range(batch_size):
                for beam in beams[i]:
                    if beam["tokens"][0, -1].item() != eos_token_id:
                        finished = False
                        break
                if not finished: break
            if finished: break

    final_captions = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    
    for i in range(batch_size):
        best_beam = sorted(beams[i], key=lambda x: x["score"], reverse=True)[0]
        generated_tokens = best_beam["tokens"][0] # این تانسور شامل BOS و کلمات تولید شده است
        
        length = min(len(generated_tokens), max_len)
        final_captions[i, :length] = generated_tokens[:length]

    return final_captions

