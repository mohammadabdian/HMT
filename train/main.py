import json
import torch

from paths import results_dir
from dataloaders import build_dataloaders
from engine import build_model, train_loop
from evaluate import evaluate_bleu
#from metrics import compute_metrics


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")
    
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download("punkt")

    epochs = 14
    

    tokenizer, train_loader, val_loader, test_loader = \
        build_dataloaders()

    model, criterion, optimizer, scheduler = \
        build_model(
            tokenizer=tokenizer,
            device=device,
            epochs=epochs,
            train_loader_len=len(train_loader)
        )
    
    print("Embedding shape:", model.embed.weight.shape)        
    print("Output Linear shape:", model.out.weight.shape)       
    print("img_map shape:", model.img_map.weight.shape)        

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
        save_dir=results_dir
    )
    """
    best_model_path = results_dir / "best_model.pt"
    last_model_path = results_dir / "last_model.pt"
    
    if best_model_path.exists():
        print("\nLoading best model for evaluation...")
        model.load_state_dict(
            torch.load(best_model_path, map_location=device))
    else:
        print("\nBest model not found. Loading last model...")
        if last_model_path.exists():
            model.load_state_dict(torch.load(last_model_path, map_location=device))
        else:
            print("No model found for evaluation!")
            return
    
    model.to(device)
    model.eval()
    
    print("\nRunning BLEU evaluation...")
    bleu_results = evaluate_bleu(
        model,
        test_loader,
        tokenizer,
        device
    )
    
    print("\nRunning advanced metrics...")
    metrics_results = compute_metrics(
        model,
        test_loader,
        tokenizer,
        device
    )
    
    all_results = {**bleu_results, **metrics_results}
    
    results_path = results_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
