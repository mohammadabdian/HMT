import torch
from torch.utils.data import DataLoader
from transformers import CLIPTokenizer
from dataset import COCOCaptionDataset
from paths import train_json, val_json, test_json
from paths import train_features, val_features, test_features


def build_dataloaders(batch_size=32, max_length=48, num_workers=8):

    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch32",
        local_files_only=True
    )

    train_dataset = COCOCaptionDataset(
        str(train_json),
        tokenizer,
        train_features,
        max_length=max_length
    )
    
    val_dataset = COCOCaptionDataset(
        str(val_json),
        tokenizer,
        val_features,
        max_length=max_length
    )

    test_dataset = COCOCaptionDataset(
        str(test_json),
        tokenizer,
        test_features,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return tokenizer, train_loader, val_loader, test_loader