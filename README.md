# Robust Pothole Avoidance in Autonomous Driving
### Using Multi-Task Transformer Perception and Ensemble Reinforcement Learning

> **⚠️ IMPORTANT DISCLAIMER**
> This implementation is a **software-only, dataset/simulation prototype** and does **not** control a physical vehicle.
> No physical camera, Raspberry Pi, Arduino, LiDAR, or real autonomous vehicle is required or used.

---

## 📌 Project Objective

End-to-end software pipeline demonstrating:

```
Pothole Image (dataset)
       ↓
Multi-Task TransUNet (Segmentation + Depth + Uncertainty)
       ↓
UASA Feature Extraction → 117-Dimensional Risk-Aware State
       ↓
PPO + A2C + TRPO + Recurrent PPO (Soft-Voting Ensemble)
       ↓
Safety Refinement Layer (SRL)
       ↓
Simulated Vehicle Decision:
  Maintain Lane / Shift Left / Shift Right / Emergency Brake
```

---

## 🏗️ Architecture

| Module | Description |
|---|---|
| `models/transunet.py` | ResNet-50 CNN + Transformer encoder + U-Net decoder |
| `models/depth_head.py` | Per-pixel depth estimation head |
| `models/uncertainty_head.py` | MC-Dropout uncertainty head |
| `models/loss.py` | Multi-task loss (BCE+Dice / SILog / Entropy regularisation) |
| `state/uasa.py` | 117-dimensional risk-aware state generator |
| `state/severity.py` | Pothole severity scorer (configurable weights) |
| `state/temporal_smoothing.py` | EMA temporal smoother |
| `rl/environment.py` | Gymnasium environment (stub/dataset/CARLA-ready) |
| `rl/ppo_agent.py` | PPO (Stable-Baselines3) |
| `rl/a2c_agent.py` | A2C (Stable-Baselines3) |
| `rl/trpo_agent.py` | Custom TRPO (conjugate gradient + line search) |
| `rl/recurrent_ppo.py` | Recurrent PPO/LSTM (sb3-contrib) |
| `rl/ensemble.py` | Soft-voting ensemble (configurable weights) |
| `safety/srl.py` | Deterministic 7-check Safety Refinement Layer |
| `safety/collision_checker.py` | Trajectory–pothole collision checking |
| `safety/trajectory.py` | Short-horizon kinematic trajectory predictor |
| `simulation/road.py` | OpenCV top-down road renderer |
| `simulation/vehicle.py` | Simulated vehicle state manager |
| `app/main.py` | Streamlit GUI application |
| `app/pipeline.py` | End-to-end inference pipeline |
| `app/demo_images.py` | Synthetic demo image generator |

---

## 📦 Installation

### 1. Prerequisites
- Python 3.12
- (Optional) CUDA-capable GPU for faster training

### 2. Create a virtual environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -e .
# or
pip install -r requirements.txt
```

---

## 📱 Real-Time Mobile Testing & HUD

Test the pothole avoidance AI in real time on your smartphone (Android / iOS):

```bash
python run_mobile.py
```

1. Ensure your smartphone and PC are connected to the same Wi-Fi or Mobile Hotspot.
2. Scan the **QR code** printed in the terminal with your phone.
3. Open the link in Chrome or Safari and tap **"START REAL-TIME HUD"**.
4. Point your camera at roads, dashcam videos, or pothole photos for instant AR overlays, dynamic steering action arrows, voice audio alerts, and vibration feedback!

See [MOBILE_TESTING_GUIDE.md](MOBILE_TESTING_GUIDE.md) for full details.

---

## 🚀 Running the Streamlit Desktop Dashboard

```bash
streamlit run app/main.py
```

Then open: http://localhost:8501

**No dataset required** — the app starts in **DEMO MODE** with synthetic pothole images.

### Quick start:
1. Open the app
2. Select "Demo Images" or "📷 Live Camera" in the sidebar
3. Click **"Run Full Pipeline"**
4. View segmentation, depth, uncertainty, UASA state, RL decisions, SRL verdict, and vehicle trajectory

---

## 📂 Dataset Setup

### Pothole-600
```bash
export POTHOLE600_ROOT=/path/to/pothole-600
```
Structure:
```
pothole-600/
├── images/    (RGB images)
└── masks/     (binary segmentation masks)
```

### PotholeRGBD
```bash
export POTHOLE_RGBD_ROOT=/path/to/pothole-rgbd
```
Structure:
```
pothole-rgbd/
├── rgb/
├── masks/
└── depth/
```

Update `configs/dataset_config.yaml`:
```yaml
datasets:
  active: "pothole600"   # or "pothole_rgbd" or "combined"
```

---

## 🧠 Training the TransUNet

```bash
python scripts/train_perception.py \
  --model_cfg configs/model_config.yaml \
  --train_cfg configs/training_config.yaml \
  --dataset_cfg configs/dataset_config.yaml
```

Checkpoints are saved to `checkpoints/perception/`.

Load the best checkpoint in the GUI by entering the path in the sidebar.

---

## 🎮 Training RL Agents

```bash
python scripts/train_rl.py \
  --rl_cfg configs/rl_config.yaml \
  --severity_cfg configs/severity_config.yaml \
  --srl_cfg configs/srl_config.yaml \
  --agent ppo    # or a2c / trpo / recurrent_ppo / all
```

---

## 📊 Running Evaluation

```bash
python scripts/evaluate_ensemble.py
python scripts/evaluate_srl.py
python scripts/run_ablation.py
```

---

## 🗃️ Understanding the 117-D State Vector

The UASA state vector always contains exactly **117 features**:

| Index | Description |
|---|---|
| 0 | Frame uncertainty (mean) |
| 1 | Frame uncertainty (max) |
| 2 | Pothole coverage ratio |
| 3 | Normalised pothole count |
| 4 | Lane centre offset |
| 5 | Mean road depth |
| 6 | Max severity |
| 7 | Mean severity |
| 8 | Global min depth (norm) |
| 9 | Aspect ratio |
| 10–26 | Pothole 1: 17 features (centroid, bbox, depth, severity, uncertainty, shape) |
| 27–43 | Pothole 2 |
| 44–60 | Pothole 3 |
| 61–77 | Pothole 4 |
| 78–94 | Pothole 5 |
| 95–116 | Temporal/EMA features |

Empty slots are zero-padded. The dimension is **always exactly 117**.

---

## 🗳️ Understanding Soft Voting

**Equal-weight formulation** (implementation choice):

```
P_ensemble(a|s) = (P_PPO + P_A2C + P_TRPO + P_RPPO) / 4
a_RL = argmax_a P_ensemble(a|s)
```

The research paper describes soft voting but does not specify exact weights.
Configurable in `configs/rl_config.yaml → ensemble.weights`.

---

## 🛡️ Understanding SRL

The SRL is **deterministic** and **independent of the RL policy**.

It checks 7 criteria:
1. Perception uncertainty ≤ threshold
2. Pothole severity ≤ threshold
3. Lateral deviation ≤ threshold
4. Lane boundary compliance
5. Trajectory–pothole collision check
6. Action consistency (oscillation detection)
7. Emergency brake condition (extreme severity + close obstacle)

If any check fails → override with safe fallback action.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Expected: 30+ tests covering UASA (117-D guarantee), severity, EMA, reward, ensemble, trajectory, collision, and SRL.

---

## 📈 Reference Results (from Research Report)

> These values are **reported by the research paper** — they are NOT reproduced by this local prototype.

| Metric | Value |
|---|---|
| Dice (CV mean) | 0.8393 ± 0.0148 |
| Final Dice | 0.8356 |
| IoU | 0.7344 |
| Depth RMSE | 4.1570 |
| Full-system lateral deviation | 0.289 m |
| RL-only baseline | 0.392 m |
| Improvement | 26.3% |
| SRL intervention rate | 55.9% |
| Inference speed | ≈ 22 FPS (16-GB GPU) |

---

## ⚠️ Limitations

1. **No trained weights included** — the app starts in DEMO mode (classical CV mock)
2. **Simulation only** — no real vehicle, no real road
3. **Pseudo-depth** — MiDaS v3 output is relative, not metric
4. **RL agents in demo** — use mock distributions, not trained policies
5. **No CARLA integration** — RL environment uses stub/dataset mode

---

## 🔭 Future Real-Camera Integration

To connect a real camera or CARLA:
1. Replace `app/pipeline.py::_demo_perception()` with a live camera feed
2. Replace `rl/environment.py::_get_observation()` with CARLA sensor queries
3. Load a trained TransUNet checkpoint via the sidebar
4. Load trained RL agent checkpoints via `scripts/train_rl.py`
5. Replace `simulation/vehicle.py` with actual vehicle actuation commands

---

## 📁 Project Structure

```
project/
├── app/                    # Streamlit GUI + pipeline
├── configs/                # All YAML configuration files
├── data/                   # Dataset interface, augmentation, pseudo-depth
├── models/                 # TransUNet, heads, multi-task loss
├── perception/             # Trainer, evaluator, inference
├── state/                  # UASA, severity, EMA smoothing
├── rl/                     # 4 RL agents, environment, ensemble
├── safety/                 # SRL, collision checker, trajectory
├── simulation/             # Road renderer, vehicle state
├── evaluation/             # Metrics accumulator
├── tests/                  # Unit tests (pytest)
├── scripts/                # Training & evaluation scripts
├── requirements.txt
├── setup.py
└── README.md
```
