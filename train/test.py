import csv
import torch
import json
from pathlib import Path

from paths import results_dir, test_json
from dataloaders import build_dataloaders
from engine import build_model
from generate import generate_caption


def run_test(num_samples=100):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    tokenizer, _, _, test_loader = build_dataloaders()

    model, _, _, _ = build_model(
        tokenizer=tokenizer,
        device=device,
        epochs=1,
        train_loader_len=1
    )

    best_model_path = results_dir / "best_model.pt"
    last_model_path = results_dir / "last_model.pt"

    if best_model_path.exists():
        model.load_state_dict(
            torch.load(best_model_path, map_location=device)
        )
        print("Loaded best model")
    elif last_model_path.exists():
        model.load_state_dict(
            torch.load(last_model_path, map_location=device)
        )
        print("Loaded last model")
    else:
        print("No model found!")
        return

    model.to(device)
    model.eval()

    output_path = results_dir / "test_results.csv"

    with open(output_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "real_caption", "generated_caption"])

        count = 0

        for image_features, captions, image_ids in test_loader:

            image_features = image_features.to(device)
            captions = captions.to(device)

            generated = generate_caption(
                model,
                image_features,
                tokenizer
            )

            batch_size = image_features.size(0)

            for i in range(batch_size):

                if count >= num_samples:
                    print(f"Saved {count} samples to {output_path}")
                    return

                img_id = image_ids[i].item()

                real_caption = tokenizer.decode(
                    captions[i].tolist(),
                    skip_special_tokens=True
                )

                generated_caption = tokenizer.decode(
                    generated[i].tolist(),
                    skip_special_tokens=True
                )

                print("=" * 50)
                print(f"Image ID: {img_id}")
                print(f"Real: {real_caption}")
                print(f"Generated: {generated_caption}")

                writer.writerow([
                    img_id,
                    real_caption,
                    generated_caption
                ])

                count += 1

    print(f"Saved {count} samples to {output_path}")


if __name__ == "__main__":
    print("Test started")
    run_test(50)
    print("Test finished successfully")