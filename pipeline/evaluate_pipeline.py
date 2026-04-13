"""
evaluate_pipeline.py
=====================
Two-stage YOLOv8 + EfficientAD inspection pipeline — VisA chewinggum.

Stage 1 — YOLOv8 detects each chewing-gum instance (or use --use_full_image)
Stage 2 — Each crop: padded → Otsu-refined → aspect-ratio resize + black pad → denoised → EfficientAD scored
Final score = max over all detected instances per image.

Typical usage — original VisA dataset (YOLO crops at inference time):
    python evaluate_pipeline.py \
        --dataset_path /path/to/VisA_20220922 \
        --ckpt_dir     ./ckptSmall \
        --yolo_weights /home/ahmed/framework/runs/detect/yolo_runs/chewinggum_finetune/weights/best.pt \
        --save_dir     ./results \
        --model_size   S \
        --device       cuda

Typical usage — pre-cropped dataset (images already 256×256, skip YOLO):
    python evaluate_pipeline.py \
        --dataset_path ./cropped_dataset \
        --ckpt_dir     ./ckpt_cropped \
        --save_dir     ./results_cropped \
        --use_full_image \
        --model_size   S \
        --device       cuda
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from load_efficientad import load_efficientad_model
from pipeline         import InspectionPipeline
from data_loader      import get_AD_dataset
from metrics          import compute_all_metrics, print_results, save_quantitative_txt
from visualisation    import (save_roc_curve, save_score_histogram,
                               save_overlay, save_tiff,
                               save_per_image_csv, print_per_image_table)

IMAGE_SIZE = 256


# ── args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Two-stage YOLOv8 + EfficientAD evaluation pipeline.')
    p.add_argument('--dataset_path',   required=True,
                   help='VisA root or cropped_dataset root '
                        '(must contain split_csv/1cls.csv)')
    p.add_argument('--ckpt_dir',       required=True,
                   help='EfficientAD checkpoint directory')
    p.add_argument('--yolo_weights',   default=None,
                   help='Fine-tuned YOLOv8 .pt weights '
                        '(ignored when --use_full_image is set)')
    p.add_argument('--category',       default='chewinggum')
    p.add_argument('--save_dir',       default='./results')
    p.add_argument('--model_size',     default='S', choices=['S', 'M'])
    p.add_argument('--device',         default=None,
                   help='cuda or cpu (auto-detected if omitted)')
    p.add_argument('--ratio',          type=float, default=0.1,
                   help='EfficientAD normalisation ratio (default 0.1)')
    p.add_argument('--conf_threshold', type=float, default=0.25,
                   help='YOLO confidence threshold (default 0.25)')
    p.add_argument('--iou_threshold',  type=float, default=0.45,
                   help='YOLO NMS IoU threshold (default 0.45)')
    p.add_argument('--pad_ratio',      type=float, default=0.02,
                   help='Context padding around each YOLO crop (default 0.02, '
                        'matches build_crop_dataset.py)')
    p.add_argument('--no_denoise',     action='store_true',
                   help='Disable Gaussian+bilateral denoising on crops')
    p.add_argument('--use_full_image', action='store_true',
                   help='Skip YOLO — use full image as one crop. '
                        'Use this when evaluating a pre-cropped dataset.')
    p.add_argument('--no_overlays',    action='store_true',
                   help='Skip saving heatmap overlay PNGs (faster)')
    p.add_argument('--no_tiffs',       action='store_true',
                   help='Skip saving float32 TIFF anomaly maps')
    p.add_argument('--n_boot',         type=int, default=500,
                   help='Bootstrap resamples for metric std (default 500)')
    return p.parse_args()


# ── main evaluation ───────────────────────────────────────────────────────────

def evaluate(args) -> dict:
    import torch

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\n{"="*60}')
    print(f'  Two-Stage Inspection Pipeline — {args.category.upper()}')
    print(f'{"="*60}')
    print(f'  Device      : {device}')
    print(f'  EfficientAD : PDN-{args.model_size}  ckpt={args.ckpt_dir}')
    if args.use_full_image:
        print(f'  Detector    : FULL-IMAGE (--use_full_image)')
    else:
        print(f'  Detector    : YOLOv8  weights={args.yolo_weights or "yolov8n.pt"}')
    print(f'  Crop padding: {args.pad_ratio * 100:.0f}%')
    print(f'  Denoising   : {"OFF" if args.no_denoise else "ON (Gaussian + Bilateral)"}')
    print(f'  Dataset     : {args.dataset_path}')
    print()

    # ── 1. Load EfficientAD ───────────────────────────────────────────────────
    models = load_efficientad_model(
        ckpt_dir   = args.ckpt_dir,
        category   = args.category,
        model_size = args.model_size,
        device     = device,
    )

    # ── 2. Build pipeline ─────────────────────────────────────────────────────
    pipeline = InspectionPipeline(
        models         = models,
        device         = device,
        use_full_image = args.use_full_image,
        detector_path  = args.yolo_weights,
        conf_threshold = args.conf_threshold,
        iou_threshold  = args.iou_threshold,
        ratio          = args.ratio,
        pad_ratio      = args.pad_ratio,
        denoise        = not args.no_denoise,
    )
    print(f'[Pipeline] YOLO active : {pipeline.detector.is_yolo_active}')

    # ── 3. Load dataset ───────────────────────────────────────────────────────
    # When --use_full_image is set the pipeline skips YOLO and sends the loaded
    # image directly to EfficientAD, so the resize here IS the only resize.
    # Use aspect-ratio-preserving resize + black padding (mirrors
    # resize_with_padding in build_crop_dataset.py) so inference crops match
    # the training crops exactly.
    # When YOLO is active the pipeline re-crops from pil_image (sample['origin'])
    # at full resolution, so this transform only affects the tensor fed to the
    # data-loader; the pipeline ignores it and works from the PIL directly.
    def _resize_with_padding(pil_img: Image.Image,
                             size: int = IMAGE_SIZE) -> Image.Image:
        """Aspect-ratio-preserving resize + black padding — mirrors build_crop_dataset.py."""
        ow, oh = pil_img.size
        scale  = size / max(ow, oh)
        nw, nh = int(ow * scale), int(oh * scale)
        resized = pil_img.resize((nw, nh), Image.LANCZOS)
        canvas  = Image.new('RGB', (size, size), (0, 0, 0))
        canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
        return canvas

    tf = transforms.Compose([
        transforms.Lambda(_resize_with_padding),
        transforms.ToTensor(),
    ])

    print(f'[Pipeline] Loading {args.category} test split ...')
    dataset = get_AD_dataset(
        type       = 'VisA',
        root       = args.dataset_path,
        transform  = tf,
        gt_transform = tf,
        phase      = 'test',
        category   = args.category,
    )

    n_normal  = sum(1 for i in range(len(dataset)) if dataset[i]['label'] == 0)
    n_anomaly = sum(1 for i in range(len(dataset)) if dataset[i]['label'] == 1)
    print(f'[Pipeline] {len(dataset)} test images  '
          f'(normal={n_normal}, anomaly={n_anomaly})')

    # ── 4. Output dirs ────────────────────────────────────────────────────────
    cat_dir     = os.path.join(args.save_dir, args.category)
    overlay_dir = os.path.join(cat_dir, 'overlays')
    map_dir     = os.path.join(cat_dir, 'anomaly_maps')
    os.makedirs(overlay_dir, exist_ok=True)
    os.makedirs(map_dir,     exist_ok=True)

    # ── 5. Inference loop ─────────────────────────────────────────────────────
    print(f'\n[Pipeline] Running inference ...')
    y_true, y_score, times_s = [], [], []
    pipeline_results = []
    csv_rows = []

    for sample in tqdm(dataset, desc='  Inference'):
        pil_image = Image.fromarray(sample['origin'].astype(np.uint8))

        result = pipeline.run(
            pil_image = pil_image,
            label     = int(sample['label']),
            name      = sample['name'],
            img_type  = sample['type'],
        )

        y_true.append(result.label)
        y_score.append(result.image_score)
        times_s.append(result.inference_ms / 1000.0)
        pipeline_results.append(result)

        if not args.no_overlays:
            save_overlay(
                pil_image, result.anomaly_map,
                os.path.join(overlay_dir,
                             f'{result.img_type}_{result.name}_overlay.png'))

        if not args.no_tiffs:
            class_map_dir = os.path.join(map_dir, result.img_type)
            os.makedirs(class_map_dir, exist_ok=True)
            save_tiff(result.anomaly_map,
                      os.path.join(class_map_dir, result.name + '.tiff'))

        csv_rows.append({
            'defect_class': result.img_type,
            'filename':     result.name,
            'gt':           result.label,
            'pred':         1 if result.prediction == 'ANOMALY' else 0,
            'score':        round(result.image_score, 6),
            'correct':      int(result.correct) if result.correct is not None else '',
            'time_ms':      round(result.inference_ms, 4),
        })

    # ── 6. Metrics ────────────────────────────────────────────────────────────
    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    times_s = np.array(times_s)

    print(f'\n[Pipeline] Computing metrics ...')
    m = compute_all_metrics(y_true, y_score, times_s, n_boot=args.n_boot)

    pipeline.set_threshold(m['threshold'])
    for r in pipeline_results:
        r.prediction = 'ANOMALY' if r.image_score >= m['threshold'] else 'NORMAL'

    # ── 7. Save outputs ───────────────────────────────────────────────────────
    print_results(m, category=args.category)
    print_per_image_table(pipeline_results)

    save_roc_curve(
        m['fpr'], m['tpr'], m['auroc'], args.category,
        os.path.join(cat_dir, 'roc_curve.png'))
    save_score_histogram(
        y_score, y_true, m['threshold'], args.category,
        os.path.join(cat_dir, 'score_histogram.png'))
    save_per_image_csv(
        csv_rows,
        os.path.join(cat_dir, 'per_image_results.csv'))

    txt_path = os.path.join(args.save_dir, 'quantitative.txt')
    save_quantitative_txt({args.category: m}, txt_path)

    # Write a single-float threshold file for single_photo_inference.py
    thr_path = os.path.join(args.save_dir, 'threshold.txt')
    with open(thr_path, 'w') as _f:
        _f.write(f"{m['threshold']:.8f}\n")

    print(f'\n[Pipeline] Results saved to {args.save_dir}/')
    print(f'  ROC curve        → {os.path.join(cat_dir, "roc_curve.png")}')
    print(f'  Score histogram  → {os.path.join(cat_dir, "score_histogram.png")}')
    print(f'  Per-image CSV    → {os.path.join(cat_dir, "per_image_results.csv")}')
    print(f'  Quantitative txt → {txt_path}')
    print(f'  Threshold file   → {thr_path}  '
          f'(pass to single_photo_inference.py --threshold_file)')

    return m


if __name__ == '__main__':
    evaluate(parse_args())