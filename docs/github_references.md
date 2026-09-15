# Open-Source Repository References and Attribution

This document details the open-source repositories and architectural references studied during the conception and engineering of **PotholeGuard-AI**.

---

## 1. Studied Repositories

### 1. `PeterHdd/pothole-detection-yolo`
- **Focus:** YOLOv5-based pothole object detection from dashcam video.
- **License:** MIT License.
- **Aspects Studied:** Real-time OpenCV video stream decoding, bounding box postprocessing.
- **Independent Implementation:** Replaced bounding box detector with Multi-Task TransUNet (pixel-level segmentation + relative depth + uncertainty).

### 2. `tamaraw01/Pothole-Segmentation`
- **Focus:** Semantic segmentation of road potholes using U-Net.
- **License:** Apache 2.0.
- **Aspects Studied:** Binary Dice loss formulation and mask contour extraction.
- **Independent Implementation:** Implemented ViT Transformer bottleneck, multi-task heads, and temporal EMA tracking.

### 3. `ArnavMandal/rgbd-pavement-segmentation`
- **Focus:** RGB-D pavement and defect segmentation.
- **License:** MIT License.
- **Aspects Studied:** Depth map alignment and relative depression calculation.
- **Independent Implementation:** Monocular depth estimation via multi-task head and camera geometry calibration formulas.

### 4. `SamaIsmail91/Road-Damage-Detection`
- **Focus:** Multi-category road damage detection (RDD2020 / RDD2022).
- **License:** MIT License.
- **Aspects Studied:** Data augmentation techniques for harsh road lighting and shadows.
- **Independent Implementation:** Specialized rain simulation, glare addition, and motion blur transforms in `datasets/transforms.py`.

---

## 2. Research Integrity & Originality Statement
- All experimental pipelines, risk assessment algorithms, deterministic safety refinement layers, and mobile HUD interfaces in PotholeGuard-AI were **independently engineered**.
- Baseline metrics cited from external benchmark publications are explicitly tagged as **"Source Reference Results"** and separated from our empirical experimental measurements.
