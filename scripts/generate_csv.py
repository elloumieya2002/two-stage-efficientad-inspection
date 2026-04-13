"""
generate_csv.py
===============
Generates split_csv/1cls.csv for MyCategory from the actual folder layout:

    MyCategory/
    └── Data/
        ├── Images/
        │   ├── Normal/     ← normal images
        │   └── Anomaly/    ← anomaly images
        └── Masks/
            └── Anomaly/    ← ground-truth masks (optional)

80% of Normal images → train split
Remaining Normal + all Anomaly → test split
"""

import csv, random
from pathlib import Path

ROOT      = Path("/home/ahmed/framework")
CATEGORY  = "MyCategory"
OUT_CSV   = ROOT / "split_csv" / "1cls.csv"
TRAIN_RATIO = 0.8
SEED        = 42

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

img_normal  = sorted((ROOT / CATEGORY / "Data" / "Images" / "Normal").glob("*"))
img_normal  = [p for p in img_normal if p.suffix.lower() in {".png",".jpg",".jpeg"}]

img_anomaly = sorted((ROOT / CATEGORY / "Data" / "Images" / "Anomaly").glob("*"))
img_anomaly = [p for p in img_anomaly if p.suffix.lower() in {".png",".jpg",".jpeg"}]

mask_dir    = ROOT / CATEGORY / "Data" / "Masks" / "Anomaly"

print(f"Normal images  : {len(img_normal)}")
print(f"Anomaly images : {len(img_anomaly)}")

random.seed(SEED)
random.shuffle(img_normal)
n_train = int(len(img_normal) * TRAIN_RATIO)
train_normal = img_normal[:n_train]
test_normal  = img_normal[n_train:]

rows = []

# train — normal only
for img in train_normal:
    rows.append([CATEGORY, "train", "normal", str(img.relative_to(ROOT)), ""])

# test — normal
for img in test_normal:
    rows.append([CATEGORY, "test", "normal", str(img.relative_to(ROOT)), ""])

# test — anomaly (look for matching mask)
for img in img_anomaly:
    mask_path = mask_dir / img.name
    mask_rel  = str(mask_path.relative_to(ROOT)) if mask_path.exists() else ""
    rows.append([CATEGORY, "test", "anomaly", str(img.relative_to(ROOT)), mask_rel])

with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["category", "split", "label", "image_path", "mask_path"])
    w.writerows(rows)

print(f"\nCSV written → {OUT_CSV}")
print(f"  train (normal)  : {len(train_normal)}")
print(f"  test  (normal)  : {len(test_normal)}")
print(f"  test  (anomaly) : {len(img_anomaly)}")
print(f"  total rows      : {len(rows)}")