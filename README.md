# two-stage-efficientad-inspection
Two-stage anomaly detection pipeline: YOLOv8 detection + EfficientAD.  Extract frames with FFmpeg → Annotate in Label Studio → Train YOLO → Crop objects → Train EfficientAD on cropped dataset → Real-time video inference with ByteTrack and DeepOCSORT. Industrial quality inspection use case (chewing gum).
