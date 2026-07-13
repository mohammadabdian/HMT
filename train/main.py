import json
import torch

from paths import results_dir
from dataloaders import build_dataloaders
from engine import build_model, train_loop
from evaluate import FastEvaluator
from generator import CaptionGenerator, GenerationConfig


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    epochs = 12

    tokenizer, train_loader, val_loader, test_loader = build_dataloaders()

    model, criterion, optimizer, scheduler = build_model(
        tokenizer=tokenizer,
        device=device,
        epochs=epochs,
        train_loader_len=len(train_loader),
    )

    print("Embedding shape:      ", model.embed.weight.shape)
    print("Output Linear shape:  ", model.out.weight.shape)
    print("img_map shape:        ", model.img_map.weight.shape)

    if model.blocks:
        print("First decoder Mamba in_proj shape:", model.blocks[0].seq_block.in_proj.weight.shape)
    else:
        print("No decoder blocks found!")

    """
    train_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        epochs=epochs,
        save_dir=results_dir,
    )
    """

    best_model_path = results_dir / "best_model.pt"
    last_model_path = results_dir / "last_model.pt"

    if best_model_path.exists():
        print("\nLoading best model for evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    elif last_model_path.exists():
        print("\nBest model not found. Loading last model...")
        model.load_state_dict(torch.load(last_model_path, map_location=device))
    else:
        print("No saved model found for evaluation!")
        return

    model.to(device)
    model.eval()

    generator = CaptionGenerator(model, tokenizer)

    features, captions, img_ids = next(iter(test_loader))
    features = features.to(device).float()

    config_beam3 = GenerationConfig(beam_size=3, max_len=48, temperature=1, use_sampling=False, top_p=1,repetition_penalty = 1)
    config_beam5 = GenerationConfig(beam_size=5, max_len=48, temperature=1, use_sampling=False, top_p=1,repetition_penalty =1)

    out1 = generator.generate_batch(features[:2], config_beam1)
    out3 = generator.generate_batch(features[:2], config_beam3)

    print("\nSample outputs:")
    print("Greedy:", tokenizer.decode(out1[0], skip_special_tokens=True))
    print("Beam-3:", tokenizer.decode(out3[0], skip_special_tokens=True))

    print("\nRunning evaluation...")
    eval_config = GenerationConfig(beam_size=3, max_len=48, temperature=1, use_sampling=False, top_p=1,repetition_penalty = 1)
    evaluator = FastEvaluator(
        model=model,
        tokenizer=tokenizer,
        eval_size=5000,
        save_dir=str(results_dir / "evaluation"),
        num_workers=4,
    )
    metrics = evaluator.evaluate(test_loader, config=eval_config)

    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
