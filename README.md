# Two-Stage EfficientAD Inspection System

> **Unsupervised Anomaly Detection** for industrial quality inspection using a YOLOv8 detection stage followed by EfficientAD anomaly scoring, with real-time video tracking support.

Inspired by:
- **EfficientAD**: *EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies* (Batzner et al., 2023)
- **Two-Stage Detection Pipeline**: Custom adaptation combining YOLOv8 object localisation with patch-distribution normalisation anomaly scoring

---

## Project Overview

This system was built for multi-instance unsupervised anomaly detection on a custom chewing-gum production dataset:

1. **Video recording** of the production line → frame extraction → YOLOv8 training dataset
2. **YOLOv8 fine-tuning** for multi-object detection (each gum piece as a separate instance)
3. **Crop dataset creation** using YOLO detections, with Otsu refinement and denoising, formatted as a VisA-compatible dataset
4. **EfficientAD retraining** (Student + AutoEncoder only, teacher frozen) on the cropped dataset
5. **Offline evaluation** with full metric suite (AUROC, AUPRO, AP, confusion matrix)
6. **Live interactive viewer** for per-image inspection
7. **Real-time video inference** with two multi-object tracking backends: ByteTrack and DeepOCSORT

---

## Repository Structure

```
two-stage-efficientad-inspection/
│
├── pipeline/                         # Main inference and evaluation scripts
│   ├── pipeline.py                   # Core InspectionPipeline class (two-stage logic)
│   ├── evaluate_pipeline.py          # Full offline evaluation with metrics
│   ├── evaluate_live.py              # Interactive live viewer (OpenCV window)
│   ├── single_photo_inference.py     # Test on one production-line photo
│   ├── video_inference.py            # Real-time inference + ByteTrack tracking
│   └── video_inference1.py           # Real-time inference + DeepOCSORT tracking
│
├── scripts/                          # Training and dataset utility scripts
│   ├── generate_csv.py               # Generate split_csv/1cls.csv from folder layout
│   ├── crop_dataset.py               # Crop detected objects using YOLO → PNG crops
│   ├── test_sizing.py                # Visualize the full cropping pipeline
│   ├── test_crop_sizes.py            # Trace every resize step in detail
│   ├── diagnose.py                   # Debug anomaly score separation issues
│   ├── recompute_quantiles.py        # Fix quantile mismatch after retraining
│   └── retrain_efficientad.py        # Retrain EfficientAD Student + AE on your crops
│
├── utils/                            # Reusable core modules
│   ├── models.py                     # Teacher (PDN-S/M), Student, AutoEncoder
│   ├── data_loader.py                # VisA, MVTec, MVTecLOCO dataset loaders
│   ├── detection.py                  # YOLOv8 Detector class with class filtering
│   ├── load_efficientad.py           # Load full model bundle from checkpoint dir
│   ├── efficientad_inference.py      # Pure inference engine (run_inference, heatmap)
│   ├── pipeline.py                   # preprocess_crop, PipelineResult dataclass
│   ├── metrics.py                    # AUROC, AUPRO, AP, confusion, bootstrap std
│   └── visualisation.py              # ROC curves, histograms, overlay PNGs, TIFFs
│
├── split_csv/
│   └── 1cls.csv                      # Dataset split file (train/test/normal/anomaly)
│
├── ckpt/                             # Checkpoint directory (not tracked in git)
│   ├── best_teacher.pth              # Frozen teacher weights (shared)
│   ├── chewinggum_student.pth        # Trained student weights
│   ├── chewinggum_autoencoder.pth    # Trained autoencoder weights
│   └── chewinggum_quantiles.npy      # Normalisation stats: mean, std, qa/qb
│
├── requirements.txt
└── README.md
```

---

## System Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 1 — Data Collection                                           │
│  Record production-line video → extract frames → label with YOLO   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 2 — YOLO Fine-tuning                                         │
│  Train YOLOv8 on labelled frames for per-instance detection        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 3 — Crop Dataset Creation  (crop_dataset.py)                 │
│  YOLO detects → pad bbox → Otsu refine → rotate portrait →        │
│  resize with black padding to 256×256 → Gaussian + Bilateral       │
│  denoise → save as VisA-compatible dataset tree                    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 4 — EfficientAD Retraining  (retrain_efficientad.py)        │
│  Teacher frozen → train Student + AutoEncoder on crops             │
│  Compute channel mean/std on crops → recompute quantiles           │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 5 — Evaluation  (evaluate_pipeline.py / evaluate_live.py)   │
│  YOLOv8 detects instances → crop each → EfficientAD scores each   │
│  Final score = max over all instances → AUROC / AUPRO / F1        │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STEP 6 — Deployment  (video_inference.py / video_inference1.py)  │
│  Live video → YOLO detects → ByteTrack or DeepOCSORT tracks →     │
│  EfficientAD scores each track → EMA smoothing → annotated output  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/elloumieya2002/two-stage-efficientad-inspection
cd two-stage-efficientad-inspection


# Create and activate the conda environment
conda create -n inspection python=3.10
# To activate this environment, use
conda activate inspection
# To deactivate an active environment, use
conda deactivate

# Install all dependencies in one command
pip install torch torchvision ultralytics scikit-learn tqdm opencv-python pillow tifffile matplotlib boxmot

```

---

## Dataset Format

The system expects a **VisA-compatible** folder layout with a CSV split file:

```
dataset_root/
├── split_csv/
│   └── 1cls.csv                      # Required by the data loader
│
└── chewinggum/
    ├── Data/
    │   ├── Images/
    │   │   ├── Normal/               # Good samples (used for train + test)
    │   │   └── Anomaly/              # Defective samples (test only)
    │   └── Masks/
    │       └── Anomaly/              # Ground-truth masks (optional)
```

### CSV format (`split_csv/1cls.csv`)

```
category,split,label,image_path,mask_path
chewinggum,train,normal,chewinggum/Data/Images/Normal/img001.png,
chewinggum,test,normal,chewinggum/Data/Images/Normal/img051.png,
chewinggum,test,anomaly,chewinggum/Data/Images/Anomaly/img001.png,chewinggum/Data/Masks/Anomaly/img001.png
```

---

## Commands

### 0 — train the YOLO model on the annotated dataset

```bash
 python scripts/train_yolo.py         --dataset_dir  /path/to/raw_images         --out_dir      /path/to/raw_images_split         --save_dir     ./yolo_runs         --epochs       50         --device       0
```


---


### 1 — Crop a raw image folder using YOLO

```bash
python scripts/crop_dataset.py \
    --input-dir  /path/to/raw_images/images \
    --out-dir    ./cropped_dataset \
    --model      /path/to/best.pt \
    --class-id   0 \
    --pad-ratio  0.05


# With denoising disabled:
python scripts/crop_dataset.py \
    --input-dir  /path/to/raw_images/images \
    --out-dir    ./cropped_dataset \
    --model      /path/to/best.pt \
    --no-denoise
```

Output: `cropped_dataset/crops/*.png` — all 256×256 preprocessed crops.

---
### 2 — Generate the split CSV

Run this first whenever you set up a new category folder:

```bash
# Edit generate_csv.py to set ROOT, CATEGORY, TRAIN_RATIO, then run:
python scripts/generate_csv.py
```

This splits 80% of normal images into train and keeps the rest + all anomaly images for test.

---

### 3 — Retrain EfficientAD on your crops

```bash
python -m scripts.retrain_efficientad \
  --dataset_path /path/to/MyCategory \
  --ckpt_dir ./ckpt_cropped \
  --teacher_path /path/to/best_teacher.pth \
  --model_size S \
  --iterations 70000 \
  --device cuda


# Without ImageNet regularisation (if you don't have ImageNet):
 python -m scripts.retrain_efficientad \
  --dataset_path /path/to/MyCategory \
  --ckpt_dir ./ckpt_cropped \
  --teacher_path /path/to/best_teacher.pth \
  --no_imagenet \
  --model_size S \
  --iterations 70000 \
  --device cuda
```

After training, copy the teacher into the new checkpoint directory:
```bash
cp /path/to/best_teacher.pth ./ckpt_cropped/best_teacher.pth
```

---

### 4 — Recompute quantiles (if scores don't separate)

Run this if normal and anomaly scores overlap after retraining:

```bash
python scripts/recompute_quantiles.py \
    --ckpt_dir      ./ckpt_cropped \
    --dataset_path  ./cropped_dataset \
    --yolo_weights  /path/to/yolo_best.pt \
    --category      chewinggum \
    --model_size    S \
    --device        cuda \
    --pad_ratio     0.02
```

---

### 5 — Diagnose score separation

```bash
python scripts/diagnose.py \
    --ckpt_dir      ./ckpt_cropped \
    --dataset_path  ./cropped_dataset \
    --category      chewinggum \
    --model_size    S \
    --device        cuda \
    --n_images      50
```

Prints raw map statistics before and after normalisation and flags quantile mismatches.

---

### 6 — Full offline evaluation (two-stage pipeline)

```bash

# With pre-cropped dataset (skip YOLO):
 python -m pipeline.evaluate_pipeline     --dataset_path /path/to/MyCategory     --ckpt_dir     /path/to/ckpt_cropped     --save_dir     ./results_cropped     --use_full_image --model_size   S     --device       cuda

# Skip saving overlays and TIFFs for speed:
 python -m pipeline.evaluate_pipeline     --dataset_path /path/to/MyCategory     --ckpt_dir     /path/to/ckpt_cropped     --save_dir     ./results_cropped  --no_overlays  --no_tiffs   --model_size   S     --device       cuda
```

**Outputs** saved to `./results/chewinggum/`:
- `roc_curve.png`
- `score_histogram.png`
- `per_image_results.csv`
- `overlays/` — heatmap PNGs
- `anomaly_maps/` — float32 TIFFs
- `../quantitative.txt` — full metric table
- `../threshold.txt` — optimal threshold for inference scripts

---

### 7 — Interactive live evaluation viewer

```bash
python pipeline/evaluate_live.py \
    --dataset_path  ./cropped_dataset \
    --ckpt_dir      ./ckpt_cropped \
    --category      chewinggum \
    --model_size    S \
    --device        cuda

# Load threshold from evaluate_pipeline.py output:
python pipeline/evaluate_live.py \
    --dataset_path  ./cropped_dataset \
    --ckpt_dir      ./ckpt_cropped \
    --category      chewinggum \
    --threshold_file ./results/threshold.txt \
    --auto_advance  5000 \
    --start_idx     0 \
    --device        cuda
```

**Controls:**
| Key | Action |
|-----|--------|
| `SPACE` / `→` | Next image |
| `←` | Previous image |
| `S` | Save current frame as PNG |
| `Q` / `ESC` | Quit |

---

### 8 — Single photo inference

```bash
python pipeline/single_photo_inference.py \
    --photo_path    /path/to/production_photo.jpg \
    --ckpt_dir      ./ckpt_cropped \
    --yolo_weights  /path/to/yolo_best.pt \
    --out_dir       ./single_photo_results \
    --category      chewinggum \
    --model_size    S \
    --device        cuda \
    --threshold     0.5
```

**Outputs** saved to `./single_photo_results/`:
- `full_image_annotated.png` — annotated photo with all bounding boxes and scores
- `cropped_NNN.png` — clean 256×256 crop for each detection
- `overlay_NNN.png` — EfficientAD heatmap blended on each crop
- `results.csv` — per-crop scores, timings, and predictions

---

### 9 — Real-time video inference

#### With ByteTrack (built into ultralytics)

```bash
python pipeline/video_inference.py \
    --source        /path/to/video.mp4 \
    --ckpt_dir      ./ckpt_cropped \
    --yolo_weights  /path/to/yolo_best.pt \
    --out_dir       ./video_results \
    --category      chewinggum \
    --model_size    S \
    --threshold     0.5 \
    --ema_alpha     0.3 \
    --device        cuda

# Live webcam (device index 0):
python pipeline/video_inference.py \
    --source        0 \
    --ckpt_dir      ./ckpt_cropped \
    --yolo_weights  /path/to/yolo_best.pt \
    --out_dir       ./webcam_results \
    --threshold     0.5 \
    --device        cuda

# Headless (no display window):
python pipeline/video_inference.py \
    --source        /path/to/video.mp4 \
    --ckpt_dir      ./ckpt_cropped \
    --yolo_weights  /path/to/yolo_best.pt \
    --no_display \
    --device        cuda
```

#### With DeepOCSORT (boxmot)

```bash
python pipeline/video_inference1.py \
    --source        /path/to/video.mp4 \
    --ckpt_dir      ./ckpt_cropped \
    --yolo_weights  /path/to/yolo_best.pt \
    --out_dir       ./video_results_deep \
    --category      chewinggum \
    --model_size    S \
    --threshold     0.5 \
    --ema_alpha     0.3 \
    --device        cuda
```

**Video outputs** saved to `--out_dir`:
- `output_video.mp4` — annotated video with track IDs, coloured boxes, and heatmap overlays
- `frame_scores.csv` — per-frame, per-track raw and EMA scores
- `per_id_summary.csv` — per-track anomaly statistics over the full video

---

### 10 — Debugging and visualisation tools

```bash
# Visualise the full crop pipeline on sampled images:
python scripts/test_sizing.py \
    --visa_root     /path/to/dataset \
    --yolo_weights  /path/to/yolo_best.pt \
    --out_dir       ./sizing_test \
    --category      chewinggum \
    --n_samples     20 \
    --device        cuda

# Trace every intermediate size transformation step by step:
python scripts/test_crop_sizes.py \
    --visa_root     /path/to/dataset \
    --yolo_weights  /path/to/yolo_best.pt \
    --category      chewinggum \
    --n_images      5 \
    --device        cuda
```

---

## Preprocessing Pipeline (Training = Inference)

Every crop goes through the **exact same** steps during training and inference to prevent distribution shift:

```
Raw image
    │
    ▼  YOLOv8 detection → bounding box (x1, y1, x2, y2)
    │
    ▼  Pad bbox by pad_ratio (default 5%)
    │
    ▼  Otsu thresholding → tighten crop to foreground
    │     (skipped if foreground coverage < 30%)
    │
    ▼  Rotate 90° if portrait (height > width)
    │     (fills the 256×256 canvas more efficiently)
    │
    ▼  Aspect-ratio-preserving resize + black padding → 256×256
    │     (mirrors resize_with_padding in all scripts)
    │
    ▼  Gaussian blur (3×3, σ=0.8)   ← removes resize aliasing
    │
    ▼  Bilateral filter (d=5, σColor=15, σSpace=15)
    │
    ▼  ToTensor()  →  [3, 256, 256], values in [0, 1]
    │     NO ImageNet normalisation here
    │     (teacher internally applies ImageNet norm via imagenet_norm_batch)
    │
    ▼  EfficientAD forward pass
         → Teacher (frozen PDN) → Student + AE → anomaly map → score
```

---

## EfficientAD Scoring (Algorithm 2)

```
image_tensor  →  Teacher  →  t_out  [B, 384, h, w]
              →  Student  →  s_out  [B, 768, h, w]
              →  AE       →  a_out  [B, 384, h, w]

norm_t  = (t_out - channel_mean) / channel_std

y_st    = s_out[:, :384]          # Student-Teacher branch
y_stae  = s_out[:, 384:]          # Student-AE branch

d_st    = (norm_t - y_st)²        # ST difference map
d_stae  = (a_out  - y_stae)²      # STAE difference map

m_st    = mean(d_st,   dim=C)  → upsample to 256×256
m_ae    = mean(d_stae, dim=C)  → upsample to 256×256

norm_mst = 0.1 × (m_st   - qa_st) / (qb_st - qa_st)
norm_mae = 0.1 × (m_ae   - qa_ae) / (qb_ae - qa_ae)

anomaly_map  = 0.5 × norm_mst + 0.5 × norm_mae
image_score  = max(anomaly_map)
```

---

## Checkpoint Layout

```
ckpt/
├── best_teacher.pth               # Shared frozen teacher (PDN-S or PDN-M)
├── chewinggum_student.pth         # Trained student
├── chewinggum_autoencoder.pth     # Trained autoencoder
└── chewinggum_quantiles.npy       # Dict with keys:
                                   #   mean   [384,1,1]  teacher channel mean
                                   #   std    [384,1,1]  teacher channel std
                                   #   qa_st  scalar     ST map p90
                                   #   qb_st  scalar     ST map p99.5
                                   #   qa_ae  scalar     AE map p90
                                   #   qb_ae  scalar     AE map p99.5
```

---

## Key Arguments Reference

| Argument | Default | Used in | Description |
|---|---|---|---|
| `--dataset_path` | required | eval, retrain | Path to dataset root |
| `--ckpt_dir` | required | all | Checkpoint directory |
| `--yolo_weights` | required | pipeline scripts | Path to fine-tuned YOLO `.pt` |
| `--category` | `chewinggum` | all | Dataset category name |
| `--model_size` | `S` | all | PDN size: `S` (small) or `M` (medium) |
| `--device` | auto | all | `cuda` or `cpu` |
| `--pad_ratio` | `0.02` | crop/eval | Context padding around YOLO box |
| `--threshold` | Youden J | eval/video | Decision threshold |
| `--threshold_file` | — | live/single | Load threshold from file |
| `--use_full_image` | False | evaluate_pipeline | Skip YOLO, use full image |
| `--no_denoise` | False | crop | Disable Gaussian + Bilateral denoising |
| `--ema_alpha` | `0.3` | video | EMA smoothing for per-track scores |
| `--auto_advance` | `10000` | live | Auto-advance interval in ms |
| `--no_imagenet` | False | retrain | Disable ImageNet regularisation term |
| `--iterations` | `70000` | retrain | Training iterations |

---

## References

```
@article{batzner2023efficientad,
  title   = {EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies},
  author  = {Batzner, Kilian and Heckler, Lars and König, Rebecca},
  journal = {arXiv preprint arXiv:2303.14535},
  year    = {2023}
}

@inproceedings{cao2022towards,
  title     = {Towards Total Recall in Industrial Anomaly Detection},
  author    = {Cao, Yunkang and Xu, Xiaoming and Sun, Chen and Cheng, Kang and Du, Zhuoyang and Gao, Shenghua and Shen, Weiming},
  booktitle = {CVPR},
  year      = {2022}
}
```
