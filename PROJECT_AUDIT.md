# 🛡️ PotholeGuard-AI PRO: Comprehensive System & Codebase Audit

**Date of Audit:** September 2026  
**System Version:** PotholeGuard-AI PRO v2.0 Architecture  
**Scope:** Complete Codebase, Models, Inference Pipelines, RL & Safety Layers, Backend/Frontend, Datasets, Evaluation Suites, and Deployments.

---

## 📌 Executive Summary

**PotholeGuard-AI** is a smartphone-based computer vision and rider assistance system engineered specifically for two-wheeler motorcyclists and scooter riders. The system utilizes a standard smartphone mounted on the handlebar, capturing the road surface through its rear camera, performing multi-task perception (TransUNet), estimating relative size and depth, assessing uncertainty, computing trajectory-based rider-path overlap, tracking hazard persistence, calculating multi-factor risk scores, applying deterministic Safety Refinement Layers (SRL), and delivering voice and visual alerts.

---

## A. Existing Features

| Category | Feature Name | Implementation Status | Module / File Location |
| :--- | :--- | :--- | :--- |
| **Perception** | Multi-Task TransUNet Backbone | Implemented (CNN + ViT + UNet) | [`models/transunet.py`](models/transunet.py) |
| **Perception** | Monocular Depth Estimation Head | Implemented (Relative & Pseudo-depth) | [`models/depth_head.py`](models/depth_head.py), [`vision/depth_processor.py`](vision/depth_processor.py) |
| **Perception** | Epistemic Uncertainty Estimation | Implemented (MC-Dropout Head) | [`models/uncertainty_head.py`](models/uncertainty_head.py), [`vision/uncertainty_processor.py`](vision/uncertainty_processor.py) |
| **Vision Analytics** | Morphological Shape & Boundary Analysis | Implemented (Circularity, Elongation, Complexity) | [`vision/shape_analyzer.py`](vision/shape_analyzer.py) |
| **Vision Analytics** | Relative Size & Dimension Estimator | Implemented (Pixel Area, BBox, Road Occupancy) | [`vision/size_estimator.py`](vision/size_estimator.py) |
| **Vision Analytics** | Rider Path Trajectory Estimator | Implemented (Lateral Offset, Corridor Overlap) | [`vision/path_estimator.py`](vision/path_estimator.py) |
| **Vision Analytics** | Approach Rate & Proximity Estimator | Implemented (Longitudinal growth, Speed rate) | [`vision/approach_estimator.py`](vision/approach_estimator.py) |
| **Vision Analytics** | Image Quality Assessor | Implemented (Blur/Laplacian, Glare, Contrast, Brightness) | [`vision/image_quality.py`](vision/image_quality.py) |
| **Tracking** | Multi-Object IoU + Centroid Tracker | Implemented (EMA smoothing, Persistence hits) | [`tracking/tracker.py`](tracking/tracker.py) |
| **Risk Engine** | 9-Factor Uncertainty-Aware Risk Engine | Implemented (`SAFE`, `WARNING`, `DANGER`, `UNCERTAIN`) | [`risk/risk_engine.py`](risk/risk_engine.py), [`risk/thresholds.py`](risk/thresholds.py) |
| **Explainable AI** | XAI Hazard Assessment Rationale | Implemented (Bullet-point causality generator) | [`risk/risk_explainer.py`](risk/risk_explainer.py) |
| **Safety** | Deterministic Safety Refinement Layer (SRL) | Implemented (7-rule Safety Guardian, Anti-Fatigue) | [`safety/safety_refinement.py`](safety/safety_refinement.py), [`safety/srl.py`](safety/srl.py) |
| **Safety** | Warning Deduplication & Cooldown | Implemented (Voice alert cooldown & escalation) | [`risk/warning_manager.py`](risk/warning_manager.py) |
| **State & RL** | 117-D UASA State Vector Generator | Implemented (Feature concatenation & normalization) | [`state/uasa_state.py`](state/uasa_state.py), [`state/uasa.py`](state/uasa.py) |
| **RL Ensemble** | Soft-Voting Multi-Policy Ensemble | Implemented (PPO, A2C, TRPO, RecurrentPPO) | [`rl/ensemble.py`](rl/ensemble.py) |
| **Backend** | FastAPI Asynchronous Server & WebSockets | Implemented (Frame streaming, REST APIs) | [`backend/app.py`](backend/app.py), [`backend/schemas.py`](backend/schemas.py) |
| **Database** | SQLite Detection Logger & Telemetry | Implemented (Session history, Summary stats) | [`backend/database.py`](backend/database.py) |
| **Frontend** | Mobile PWA & Canvas HUD Interface | Implemented (HTML5, JS, CSS, Canvas AR) | [`frontend/index.html`](frontend/index.html), [`frontend/app.js`](frontend/app.js) |
| **Desktop GUI** | Tkinter Inspection GUI | Implemented (avxway/pothole-detection-system style) | [`gui/app.py`](gui/app.py), [`main.py`](main.py) |
| **Streamlit** | Interactive Research & Simulation Web App | Implemented (Flagship Streamlit HUD app) | [`app/main.py`](app/main.py), [`app/pipeline.py`](app/pipeline.py) |

---

## B. Existing Models

1. **MultiTaskTransUNet ([`models/transunet.py`](models/transunet.py))**:
   - **Backbone Encoder**: ResNet-50 CNN stages extracted via `timm` (`resnet50.a1_in1k`).
   - **Transformer Bottleneck**: ViT-style Multi-Head Self-Attention (MHSA) encoder blocks with configurable embedding dimension ($D=768$) and patch embedding on deep features.
   - **Decoder**: Cascaded U-Net upsampling decoder blocks with skip connections from CNN feature stages ($C_1, C_2, C_3$).
   - **Output Heads**:
     - *Segmentation Head* ([`models/transunet.py`](models/transunet.py)): $1 \times H \times W$ Sigmoid logits for binary pothole segmentation.
     - *Depth Head* ([`models/depth_head.py`](models/depth_head.py)): Monocular relative depth regression with inverse disparity activation.
     - *Uncertainty Head* ([`models/uncertainty_head.py`](models/uncertainty_head.py)): Monte Carlo Dropout (MC-Dropout) variance estimator producing spatial epistemic uncertainty maps $[0, 1]$.
2. **Reinforcement Learning Multi-Policy Ensemble ([`rl/ensemble.py`](rl/ensemble.py))**:
   - **State Space**: 117-dimensional Unified Autonomous Safety Assessment (UASA) vector encoding perception, depth, spatial kinematics, ego dynamics, and risk indicators.
   - **Policies**: PPO, A2C, TRPO, and Recurrent-PPO (LSTM).
   - **Ensemble Aggregation**: Soft-voting probability averaging $P_{\text{ensemble}}(a|s) = \frac{1}{N}\sum_{i=1}^N P_i(a|s)$ with agreement score metrics.

---

## C. Existing Datasets & Configuration

1. **Configured Datasets ([`configs/dataset_config.yaml`](configs/dataset_config.yaml))**:
   - **Pothole-600 Benchmark**: 600 high-resolution road surface images with polygon annotations.
   - **PotholeRGBD**: Multi-modal RGB + depth ground truth dataset.
   - **Synthetic Pothole Simulation Dataset**: Generated road cavity scenes for edge-case validation.
2. **YAML Configuration Suite ([`configs/`](configs/))**:
   - `dataset_config.yaml`: Splits (Train 70%, Val 15%, Test 15%), augmentation parameters (CLAHE, Albumentations).
   - `model_config.yaml`: TransUNet architecture parameters, patch size, embedding dim, heads.
   - `risk_config.yaml` / `severity_config.yaml`: 9-factor risk weighting factors, danger thresholds.
   - `srl_config.yaml`: Safety Refinement Layer deterministic rules and override triggers.
   - `rl_config.yaml`: Hyperparameters for PPO, A2C, TRPO, and Recurrent PPO.

---

## D. Existing Working Functionality

- **Unit & Integration Test Suite**: 41/41 passing tests covering perception models, backend API, collision checking, reward functions, tracking, UASA state generation, vision extraction, SRL, and ensemble voting.
- **FastAPI Backend Server ([`scripts/run_backend.py`](scripts/run_backend.py))**: Runs on `http://0.0.0.0:8000` with WebSocket `/ws/detect`, image testing `/api/test/image`, calibration `/api/calibrate`, detection logs `/api/history`, and stats `/api/stats`.
- **Smartphone HUD ([`scripts/run_mobile_hud.py`](scripts/run_mobile_hud.py))**: Generates terminal QR code with auto-detected local IP for instant mobile Wi-Fi testing.
- **Explainable Hazard Rationale**: Every hazard detection produces bulleted rationale (e.g. *Large surface area*, *Deep depression*, *Direct path overlap*, *Stable across 8 frames*).
- **SpeechSynthesis Voice Warnings**: Web Speech API announces advisory alerts ("*Warning: Deep pothole ahead*") with a 3.0s anti-fatigue cooldown.
- **SQLite Event Logging**: Automated persistence of detection events to `potholeguard.db` with CSV export capability.

---

## E. Missing Functionality

1. **Standalone React + TypeScript Mobile Frontend**: While the HTML5/Canvas PWA HUD is functional, a modular React + TypeScript + Vite PWA frontend with service-worker offline caching and device vibration API integration will provide a native smartphone app feel.
2. **GPS Geolocation Logging**: Lat/Long coordinates are currently optional database columns; browser Geolocation API integration can automatically geo-tag potholes for hazard mapping.
3. **Automated Calibration Calibration Helper**: Interactive camera mount angle guide using smartphone accelerometer/gyroscope sensors (`DeviceOrientationEvent`).
4. **MiDaS v3 Pseudo-Depth Loader**: Explicit torch.hub teacher pipeline for training-time pseudo-depth generation when true RGB-D datasets are not present.

---

## F. Broken / Fixed Functionality

- **Resolved**: Streamlit Cloud `ImportError: libGL.so.1` fixed by migrating to `opencv-python-headless` and creating `packages.txt` containing `libgl1` and `libglib2.0-0`.
- **Resolved**: Streamlit Cloud interface synchronized to render the dark neon Rider Safety HUD natively.

---

## G. Performance Bottlenecks & Optimization

1. **Inference Latency on CPU**: Running 512x512 ResNet-50 + ViT Transformer on standard CPU takes ~40-70ms per frame. Frame sampling (e.g. 10-15 FPS) and image downscaling (480x360 for transmission) keep streaming smooth.
2. **WebSocket Image Serialization**: Base64 JSON encoding adds ~33% payload overhead. For ultra-low-latency deployment, raw binary WebSockets or WebRTC can be utilized.

---

## H. Deployment Limitations

1. **Streamlit Community Cloud Sandbox**:
   - Streamlit Cloud runs within an isolated container on port 8501 without direct exposure of custom arbitrary WebSocket ports to external mobile phones.
   - Using embedded client-side HUD execution with REST / synthetic fallback ensures 100% uptime on Streamlit Cloud, while standalone FastAPI (`scripts/run_backend.py`) serves local/VPS mobile devices.
2. **HTTPS / WSS Camera Security Requirement**:
   - Modern mobile browsers (iOS Safari, Android Chrome) strictly block camera access (`getUserMedia`) on non-HTTPS origins (except `localhost`). When hosting on a local network, a self-signed SSL certificate, ngrok tunnel, or cloud HTTPS domain (Streamlit Cloud / Render) is required.

---

## I. Recommended Upgrade Plan (Phases 1–14)

1. **Phase 1 (Complete)**: Comprehensive Project Audit (`PROJECT_AUDIT.md`).
2. **Phase 2**: Refine and standardize all configuration files (`configs/risk_config.yaml`, `configs/model_config.yaml`).
3. **Phase 3**: Verify Multi-Task TransUNet perception pipeline, monocular depth scaling, and MC-Dropout uncertainty.
4. **Phase 4**: Polish explainable 9-factor risk engine and shape extraction.
5. **Phase 5**: Multi-object temporal IoU + centroid tracking with EMA persistence.
6. **Phase 6**: Rider-path trajectory corridor projection and approach rate estimation.
7. **Phase 7**: Multimodal alert system (Voice, Visual AR, Vibration).
8. **Phase 8**: Safety Refinement Layer (SRL) deterministic validation.
9. **Phase 9**: RL multi-policy soft-voting ensemble & 117-D UASA state.
10. **Phase 10**: Modern Mobile Web/PWA interface enhancements.
11. **Phase 11**: FastAPI WebSocket & REST backend integration.
12. **Phase 12**: Streamlit research dashboard expansion.
13. **Phase 13**: Automated testing & evaluation benchmarks.
14. **Phase 14**: Academic research documentation, architecture diagrams, and final-year project report support.
