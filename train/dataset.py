import numpy as np
from torch.utils.data import Dataset
import json
import torch
import os


class COCOCaptionDataset(Dataset):
    def __init__(self, json_path: str, tokenizer, features_root, max_length):
        with open(json_path, 'r') as f:
            self.samples = json.load(f)['samples']
        
        self.tokenizer = tokenizer
        self.features_root = features_root
        self.max_length = max_length

        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if self.tokenizer.bos_token is None:
            self.tokenizer.add_special_tokens({'bos_token': '[BOS]'})
        
        new_eos_token = "[EOS_NEW]"
        self.tokenizer.add_special_tokens({'eos_token': new_eos_token})
        print(len(self.tokenizer))

    def clean_caption(self, caption):
        caption = caption.lower()
        caption = "".join([c for c in caption if c.isalpha() or c.isspace()])
        caption = " ".join(caption.split())
        return caption

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        for i in range(idx, len(self.samples)):
            try:
                sample = self.samples[i]
                img_id = sample['image_id']
                caption = sample['caption']
                caption = self.clean_caption(caption)

                feature_path = self.features_root / f"{img_id}.npz"
                features = np.load(feature_path)['feature']
                features = torch.from_numpy(features).float()

                # فقط همین، بدون هیچ دستکاری اضافه!
                tokens = self.tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                    add_special_tokens=True   # <-- این خودش همه کار رو می‌کنه
                )
                
                input_ids = tokens["input_ids"].squeeze(0)

                return features, input_ids, img_id

            except Exception:
                continue

        raise RuntimeError("No valid data found in dataset")