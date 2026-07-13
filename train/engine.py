import time
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
from model.model import Captioner, Config

# Hyperparameter settings
INITIAL_LR = 1e-4
GAMMA = 0.95

def build_model(tokenizer, device, epochs, train_loader_len):

    cfg = Config()

    model = Captioner(cfg).to(device)

    criterion = nn.CrossEntropyLoss(
        ignore_index=tokenizer.pad_token_id,
        label_smoothing=0.1
    )

    optimizer = AdamW(
        model.parameters(),
        lr=INITIAL_LR,
        weight_decay=0.01
    )

    scheduler = ExponentialLR(optimizer, gamma=GAMMA)

    return model, criterion, optimizer, scheduler


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    criterion,
    device,
    epochs,
    save_dir
):

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(epochs):

        model.train()
        epoch_loss = 0
        start_time = time.time()

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs}",
            unit="batch"
        )

        for features, tokens, _ in train_bar:

            features = features.to(device).float()
            tokens = tokens.to(device)

            input_txt = tokens[:, :-1]
            target_txt = tokens[:, 1:]

            logits = model(input_txt, features)

            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                target_txt.reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()

            train_bar.set_postfix({
                "train_loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

        # Step the scheduler after each epoch
        scheduler.step()

        avg_train_loss = epoch_loss / len(train_loader)

        model.eval()
        val_loss = 0

        with torch.no_grad():

            for features, tokens, _ in val_loader:

                features = features.to(device).float()
                tokens = tokens.to(device)

                input_txt = tokens[:, :-1]
                target_txt = tokens[:, 1:]

                logits = model(input_txt, features)

                loss = criterion(
                    logits.reshape(-1, logits.size(-1)),
                    target_txt.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        duration = time.time() - start_time

        print(
            f"\n> Epoch {epoch+1}: "
            f"Train {avg_train_loss:.4f} | "
            f"Val {avg_val_loss:.4f} | "
            f"Time {duration:.2f}s | "
            f"LR {optimizer.param_groups[0]['lr']:.6f}"
        )

        torch.save(
            model.state_dict(),
            save_dir / "last_model.pt"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

            torch.save(
                model.state_dict(),
                save_dir / "best_model.pt"
            )

            print(">>> Best model saved.")
