from paths import train_json, val_json, test_json
from paths import train_features, val_features, test_features
from transformers import CLIPTokenizer

from dataset import COCOCaptionDataset

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32",local_files_only=True)
dataset = COCOCaptionDataset(str(train_json),tokenizer,train_features,max_length=48)

fe, caption, _ = dataset.__getitem__(10)
print("features is: ", fe)
print("caption is: ", caption)
