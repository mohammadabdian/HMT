from pathlib import Path

def get_project_root():
    return Path(__file__).resolve().parents[1]

root = get_project_root()

# Features
train_features = root / "features" / "train"
val_features   = root / "features" / "val"
test_features  = root / "features" / "test"

# Dataset JSON files (COCO Karpathy split)
train_json = root / "dataset/dataset" / "train_data.json"
val_json   = root / "dataset/dataset" / "val_data.json"
test_json  = root / "dataset/dataset" / "test_data.json"

# Original COCO JSON (if needed)
original_coco_json = root / "dataset" / "mscoco2014" / "karpathy_split" / "dataset_coco.json"

# Images (COCO)
train_images_dir = root / "dataset" / "mscoco2014" / "train2014"
val_images_dir   = root / "dataset" / "mscoco2014" / "val2014"

# Results
results_dir = root / "results"
results_dir.mkdir(parents=True, exist_ok=True)