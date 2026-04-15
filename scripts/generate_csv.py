"""
generate_csv.py  –  generic CSV generator for any VisA-style custom dataset
=============================================================================

Expected layout  (mirrors what VisaDataset in data_loader.py assumes):

    <DATASET_ROOT>/                          ← --dataset_path
    ├── split_csv/
    │   └── 1cls.csv                         ← generated here
    └── <CATEGORY>/                          ← --category
        └── Data/
            ├── Images/
            │   ├── Normal/
            │   └── Anomaly/
            └── Masks/
                └── Anomaly/   (optional – masks matched by filename)

Path rules enforced by VisaDataset (data_loader.py lines 205, 228):
  split_file    = root + "/split_csv/1cls.csv"
  img_src_path  = os.path.join(root, image_path)
                  → image_path must be relative to root,
                    i.e. start with <CATEGORY>/Data/...

Usage
-----
Edit the two variables below (DATASET_ROOT and CATEGORY), then run:

    python scripts/generate_csv.py

To add a second category later, just change CATEGORY and re-run.
"""

import csv, random
from pathlib import Path

# ── USER SETTINGS  (only these two lines ever need to change) ─────────────────
DATASET_ROOT = Path("/home/ahmed/two-stage-efficientad-inspection/MyCategory")
CATEGORY     = "chewinggum"          # must match --category at training time
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_RATIO = 0.8
SEED        = 42
IMG_EXTS    = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

category_dir = DATASET_ROOT / CATEGORY
normal_dir   = category_dir / "Data" / "Images" / "Normal"
anomaly_dir  = category_dir / "Data" / "Images" / "Anomaly"
mask_dir     = category_dir / "Data" / "Masks"  / "Anomaly"
out_csv      = DATASET_ROOT / "split_csv" / "1cls.csv"

# ── sanity checks ─────────────────────────────────────────────────────────────
assert category_dir.is_dir(), f"Category folder not found: {category_dir}"
assert normal_dir.is_dir(),   f"Normal images folder not found: {normal_dir}"

# ── collect images ─────────────────────────────────────────────────────────────
img_normal  = sorted(p for p in normal_dir.glob("*")  if p.suffix.lower() in IMG_EXTS)
img_anomaly = sorted(p for p in anomaly_dir.glob("*") if p.suffix.lower() in IMG_EXTS) \
              if anomaly_dir.is_dir() else []

print(f"Dataset root   : {DATASET_ROOT}")
print(f"Category       : {CATEGORY}")
print(f"Normal images  : {len(img_normal)}")
print(f"Anomaly images : {len(img_anomaly)}")

if not img_normal:
    raise RuntimeError(f"No images found in {normal_dir}")

# ── train / test split on normal images ───────────────────────────────────────
random.seed(SEED)
shuffled     = img_normal.copy()
random.shuffle(shuffled)
n_train      = int(len(shuffled) * TRAIN_RATIO)
train_normal = shuffled[:n_train]
test_normal  = shuffled[n_train:]

# ── build rows
# image_path must be relative to DATASET_ROOT so that
# os.path.join(root, image_path) in VisaDataset resolves correctly
rows = []

for img in train_normal:
    rows.append([CATEGORY, "train", "normal", img.relative_to(DATASET_ROOT).as_posix(), ""])

for img in test_normal:
    rows.append([CATEGORY, "test", "normal", img.relative_to(DATASET_ROOT).as_posix(), ""])

for img in img_anomaly:
    mask     = mask_dir / img.name
    mask_rel = mask.relative_to(DATASET_ROOT).as_posix() if mask.exists() else ""
    rows.append([CATEGORY, "test", "anomaly", img.relative_to(DATASET_ROOT).as_posix(), mask_rel])

# ── write CSV ──────────────────────────────────────────────────────────────────
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "split", "label", "image_path", "mask_path"])
    writer.writerows(rows)

print(f"\nCSV written → {out_csv}")
print(f"  train  (normal)  : {len(train_normal)}")
print(f"  test   (normal)  : {len(test_normal)}")
print(f"  test   (anomaly) : {len(img_anomaly)}")
print(f"  total rows       : {len(rows)}")
print(f"\nSample image_path  : {rows[0][3]}")
print(f"  resolves to      : {DATASET_ROOT / rows[0][3]}")