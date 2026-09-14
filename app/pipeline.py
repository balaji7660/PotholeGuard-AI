"""
app/pipeline.py
Complete end-to-end inference pipeline (no GUI dependency).

Wires together:
  image → preprocess → TransUNet → UASA → RL ensemble → SRL → decision

In DEMO MODE (no checkpoint), returns deterministic plausible outputs
clearly labeled as such.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from omegaconf import OmegaConf, DictConfig

# --- State / Safety ---
from state.uasa import UASAStateGenerator, STATE_DIM
from state.severity import PotholeSeverityScorer, PotholeRegion
from state.temporal_smoothing import EMATemporalSmoother
from safety.srl import SafetyRefinementLayer, SRLDecision
from safety.collision_checker import CollisionChecker
from safety.trajectory import TrajectoryPredictor
from simulation.road import RoadRenderer
from simulation.vehicle import SimulatedVehicle

ACTION_NAMES = {0: "Maintain Lane", 1: "Shift Left", 2: "Shift Right"}

# Default severity config (used when severity_config.yaml is unavailable)
_DEFAULT_SEV_CFG = OmegaConf.create({
    "severity": {
        "area_weight": 0.30, "avg_depth_weight": 0.30,
        "min_depth_weight": 0.20, "lane_displacement_weight": 0.20,
        "max_area_pixels": 50000, "max_depth_metres": 0.5,
        "max_lane_displacement": 1.75, "clip_min": 0.0, "clip_max": 1.0,
        "thresholds": {"low": 0.33, "medium": 0.66, "high": 1.0},
    }
})


# Config loader
# ---------------------------------------------------------------------------

def _load_cfg(config_dir: str = None) -> dict:
    """Load all YAML configs into a merged DictConfig."""
    if config_dir is None:
        config_dir = Path(__file__).parent.parent / "configs"
    else:
        config_dir = Path(config_dir)

    cfgs = {}
    for name in ["model_config", "training_config", "rl_config", "srl_config",
                 "dataset_config", "severity_config"]:
        p = config_dir / f"{name}.yaml"
        if p.exists():
            cfgs[name] = OmegaConf.load(p)
    return cfgs


# ---------------------------------------------------------------------------
# Demo-mode perception (no trained weights needed)
# ---------------------------------------------------------------------------

def _demo_perception(image_rgb: np.ndarray) -> dict:
    """
    Produce deterministic demo perception outputs from an RGB image.
    Uses simple classical CV (Otsu threshold on grayscale) to approximate
    a pothole mask. Depth and uncertainty are synthetic.

    CLEARLY LABELED AS DEMO — not from a trained TransUNet.
    """
    H, W = image_rgb.shape[:2]

    # --- Pothole mask: Otsu threshold on grayscale (demo approximation) ---
    gray  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Keep only the bottom 2/3 of image (road region)
    road_mask = np.zeros_like(mask_raw)
    road_mask[H//3:, :] = mask_raw[H//3:, :]

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN,  kernel)

    seg = (road_mask / 255.0).astype(np.float32)

    # --- Pseudo depth: simulate with distance gradient + noise ---
    y_coords = np.linspace(1.0, 0.1, H, dtype=np.float32)  # far=near top
    depth    = np.tile(y_coords[:, np.newaxis], (1, W)) * 20.0  # 0.1..20 m
    # Add pothole "holes" (deeper in mask regions)
    depth[seg > 0.5] *= 0.08   # potholes appear closer/lower
    depth = depth.astype(np.float32)

    # --- Uncertainty: higher near edges of segmentation ---
    edge_kernel = cv2.Sobel(seg, cv2.CV_32F, 1, 1, ksize=3)
    uncertainty = np.abs(edge_kernel)
    # Add general uncertainty in pothole regions
    uncertainty[seg > 0.5] += 0.3
    uncertainty = np.clip(uncertainty, 0.0, 1.0)

    return {
        "segmentation": seg,
        "depth":        depth,
        "uncertainty":  uncertainty,
        "demo_mode":    True,
    }


# ---------------------------------------------------------------------------
# RL demo agents (deterministic mock, labeled as demo)
# ---------------------------------------------------------------------------

def _demo_rl_probs(state: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Generate demo RL action probability distributions from the UASA state.
    Uses severity/uncertainty features to bias toward avoidance.
    NOT a trained policy — implementation demo only.
    """
    severity    = float(np.clip(state[6], 0, 1))
    uncertainty = float(np.clip(state[0], 0, 1))
    centroid_x  = float(state[1]) if state[1] > 0 else 0.5

    # Base distribution — biased by where the pothole is
    if centroid_x < 0.5:          # pothole on left → prefer right
        base = np.array([0.15, 0.10, 0.75], dtype=np.float32)
    elif centroid_x > 0.5:        # pothole on right → prefer left
        base = np.array([0.15, 0.75, 0.10], dtype=np.float32)
    else:                          # centre → maintain
        base = np.array([0.60, 0.20, 0.20], dtype=np.float32)

    # Add diversity + small random perturbation (seeded by severity)
    rng = np.random.default_rng(int(severity * 1000) % 9999)
    noise = rng.uniform(-0.05, 0.05, 3).astype(np.float32)

    def make_dist(shift: np.ndarray) -> np.ndarray:
        d = np.clip(base + noise + shift, 0.01, 1.0)
        return d / d.sum()

    return {
        "ppo":           make_dist(np.array([ 0.0,  0.0,  0.0])),
        "a2c":           make_dist(np.array([-0.05, 0.05, 0.0])),
        "trpo":          make_dist(np.array([ 0.05,-0.05, 0.0])),
        "recurrent_ppo": make_dist(np.array([ 0.0,  0.0,  0.0]) + rng.uniform(-0.03, 0.03, 3)),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Complete result of one pipeline run."""
    # Perception
    original_rgb:    np.ndarray = field(repr=False)
    segmentation:    np.ndarray = field(repr=False)
    depth:           np.ndarray = field(repr=False)
    uncertainty:     np.ndarray = field(repr=False)
    demo_mode:       bool = True

    # Features
    pothole_features: List[dict] = field(default_factory=list)
    state_vector:     np.ndarray = field(default_factory=lambda: np.zeros(117))
    n_potholes:       int = 0

    # RL
    agent_probs:     Dict[str, np.ndarray] = field(default_factory=dict)
    ensemble_probs:  np.ndarray = field(default_factory=lambda: np.zeros(3))
    rl_action:       int = 0

    # SRL
    srl_decision:    Optional[SRLDecision] = None
    final_action:    int = 0

    # Simulation
    road_image:      Optional[np.ndarray] = field(default=None, repr=False)
    trajectories:    Dict[int, list] = field(default_factory=dict)
    collision_map:   Dict[int, bool]  = field(default_factory=dict)

    # Timing
    inference_ms:    float = 0.0


class InferencePipeline:
    """
    Full end-to-end inference pipeline.

    Always runs in DEMO MODE (using classical CV mock) unless a trained
    TransUNet checkpoint is loaded via load_perception_checkpoint().
    """

    def __init__(self, config_dir: str = None) -> None:
        self.cfgs = _load_cfg(config_dir)
        self._perception_model = None   # None = demo mode
        self.demo_mode = True

        rl_cfg       = self.cfgs.get("rl_config",       OmegaConf.create({}))
        severity_cfg = self.cfgs.get("severity_config",  OmegaConf.create({}))
        srl_cfg      = self.cfgs.get("srl_config",       OmegaConf.create({}))

        # UASA state generator
        try:
            self.uasa = UASAStateGenerator(severity_cfg, rl_cfg)
        except Exception:
            self.uasa = None

        # Safety components
        try:
            self.srl              = SafetyRefinementLayer(srl_cfg)
            self.collision_checker= CollisionChecker(srl_cfg)
            self.traj_pred        = TrajectoryPredictor(srl_cfg)
        except Exception:
            self.srl               = None
            self.collision_checker = None
            self.traj_pred         = None

        # Simulation
        self.road_renderer = RoadRenderer()
        self.vehicle       = SimulatedVehicle()

        # EMA smoother
        self.ema = EMATemporalSmoother(STATE_DIM, alpha=0.7)

    # ------------------------------------------------------------------

    def load_perception_checkpoint(self, ckpt_path: str) -> bool:
        """
        Attempt to load a trained TransUNet checkpoint.
        Returns True on success, False on failure (stays in demo mode).
        """
        try:
            import torch
            from models.transunet import TransUNet
            model_cfg = self.cfgs.get("model_config")
            if model_cfg is None:
                return False
            model = TransUNet(model_cfg)
            ckpt  = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt.get("model_state", ckpt))
            model.eval()
            self._perception_model = model
            self.demo_mode = False
            return True
        except Exception as e:
            return False

    # ------------------------------------------------------------------

    def run(self, image_bgr: np.ndarray) -> PipelineResult:
        """
        Run full pipeline on a BGR image.

        Parameters
        ----------
        image_bgr : np.ndarray  (H, W, 3) BGR uint8

        Returns
        -------
        PipelineResult
        """
        t0 = time.perf_counter()

        # 1. Resize & convert
        img_resized = cv2.resize(image_bgr, (512, 512))
        img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        # 2. Perception
        if self._perception_model is not None:
            perception = self._run_model_perception(img_rgb)
        else:
            perception = _demo_perception(img_rgb)

        seg = perception["segmentation"]
        dep = perception["depth"]
        unc = perception["uncertainty"]

        # 3. UASA state
        state, pothole_features = self._extract_state(seg, dep, unc)

        # 4. RL ensemble
        agent_probs = _demo_rl_probs(state)
        ensemble_probs = np.mean(list(agent_probs.values()), axis=0)
        ensemble_probs /= ensemble_probs.sum()
        rl_action = int(np.argmax(ensemble_probs))

        # 5. Trajectories & collision
        trajectories  = {}
        collision_map = {}
        if self.traj_pred and self.collision_checker:
            binary_mask = (seg >= 0.5).astype(np.uint8)
            for act in [0, 1, 2]:
                wp = self.traj_pred.predict(act, (self.vehicle.x_norm, 0.0))
                trajectories[act] = wp
                col, _ = self.collision_checker.check(wp, binary_mask)
                collision_map[act] = col

        # 6. SRL
        srl_result = None
        final_action = rl_action
        if self.srl:
            sev  = float(state[6]) if len(state) > 6 else 0.0
            unc_s= float(state[0]) if len(state) > 0 else 0.0
            min_d= float(dep.min())
            binary_mask = (seg >= 0.5).astype(np.uint8)
            srl_result   = self.srl.evaluate(
                rl_action, unc_s, sev, self.vehicle.x_norm, min_d, binary_mask
            )
            final_action = srl_result.final_action

        # 7. Update vehicle
        emergency = (srl_result is not None and
                     srl_result.override_reason is not None and
                     "EMERGENCY" in srl_result.override_reason)
        self.vehicle.apply_action(final_action, emergency=emergency)

        # 8. Road render
        potholes_for_render = self._pothole_features_to_render(pothole_features)
        road_img = self.road_renderer.render(
            vehicle_x_norm=self.vehicle.x_norm,
            vehicle_y_norm=self.vehicle.y_norm,
            potholes_norm=potholes_for_render,
            trajectories=trajectories,
            selected_action=rl_action,
            collision_map=collision_map,
            srl_override=(srl_result is not None and not srl_result.accepted),
            final_action=final_action,
            override_reason=srl_result.override_reason if srl_result else None,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return PipelineResult(
            original_rgb=img_rgb,
            segmentation=seg,
            depth=dep,
            uncertainty=unc,
            demo_mode=self.demo_mode,
            pothole_features=pothole_features,
            state_vector=state,
            n_potholes=len(pothole_features),
            agent_probs=agent_probs,
            ensemble_probs=ensemble_probs,
            rl_action=rl_action,
            srl_decision=srl_result,
            final_action=final_action,
            road_image=road_img,
            trajectories=trajectories,
            collision_map=collision_map,
            inference_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------

    def _run_model_perception(self, image_rgb: np.ndarray) -> dict:
        """Run trained TransUNet (when checkpoint is loaded)."""
        import torch
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img  = (image_rgb.astype(np.float32) / 255.0 - mean) / std
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            out = self._perception_model(tensor)
        return {
            "segmentation": out["segmentation"].squeeze().numpy(),
            "depth":        out["depth"].squeeze().numpy(),
            "uncertainty":  out["uncertainty"].squeeze().numpy(),
            "demo_mode":    False,
        }

    def _extract_state(
        self,
        seg: np.ndarray,
        dep: np.ndarray,
        unc: np.ndarray,
    ) -> Tuple[np.ndarray, List[dict]]:
        """Compute UASA state vector and extract per-pothole features for display."""
        from skimage import measure
        H, W = seg.shape
        binary = (seg >= 0.5).astype(np.uint8)
        labeled = measure.label(binary, connectivity=2)
        props   = measure.regionprops(labeled)
        props   = sorted(props, key=lambda p: p.area, reverse=True)[:5]

        pothole_features = []
        for p in props:
            region_mask = labeled == p.label
            depths_in   = dep[region_mask]
            unc_in      = unc[region_mask]
            cx_px, cy_px = p.centroid[1], p.centroid[0]
            lane_disp   = (cx_px - W / 2) / (W / 2) * 1.75
            min_row, min_col, max_row, max_col = p.bbox
            sev_scorer  = PotholeSeverityScorer(
                self.cfgs.get("severity_config") or _DEFAULT_SEV_CFG
            )
            severity = sev_scorer.score(PotholeRegion(
                area_pixels=float(p.area),
                avg_depth=float(depths_in.mean()) if len(depths_in) else 0.0,
                min_depth=float(depths_in.min())  if len(depths_in) else 0.0,
                lane_displacement=lane_disp,
            ))
            pothole_features.append({
                "centroid_x": cx_px / W,
                "centroid_y": cy_px / H,
                "bbox_w":     (max_col - min_col) / W,
                "bbox_h":     (max_row - min_row) / H,
                "area":       float(p.area),
                "area_norm":  float(p.area) / (H * W),
                "lane_displacement": lane_disp,
                "avg_depth":  float(depths_in.mean()) if len(depths_in) else 0.0,
                "min_depth":  float(depths_in.min())  if len(depths_in) else 0.0,
                "depth_var":  float(depths_in.var())  if len(depths_in) else 0.0,
                "severity":   severity,
                "uncertainty": float(unc_in.mean()) if len(unc_in) else 0.0,
            })

        # UASA state
        if self.uasa:
            try:
                state = self.uasa.compute(seg, dep, unc)
            except Exception:
                state = self._fallback_state(seg, dep, unc, pothole_features)
        else:
            state = self._fallback_state(seg, dep, unc, pothole_features)

        state = self.ema(state)
        return state, pothole_features

    @staticmethod
    def _fallback_state(seg, dep, unc, features) -> np.ndarray:
        """Build a minimal 117-d state if UASA module fails."""
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = float(unc.mean())
        state[1] = float(unc.max())
        state[2] = float((seg >= 0.5).mean())
        state[3] = float(len(features)) / 5.0
        state[6] = max((f["severity"] for f in features), default=0.0)
        state[7] = float(np.mean([f["severity"] for f in features])) if features else 0.0
        state[8] = float(dep.min()) / 80.0
        for i, f in enumerate(features[:5]):
            b = 10 + i * 17
            state[b+ 0] = 1.0
            state[b+ 1] = f["centroid_x"]
            state[b+ 2] = f["centroid_y"]
            state[b+ 3] = f["bbox_w"]
            state[b+ 4] = f["bbox_h"]
            state[b+ 5] = f["area_norm"]
            state[b+ 6] = np.clip(f["lane_displacement"] / 1.75, -1, 1)
            state[b+ 7] = f["avg_depth"] / 80.0
            state[b+ 8] = f["min_depth"] / 80.0
            state[b+10] = f["severity"]
            state[b+11] = f["uncertainty"]
        return state

    @staticmethod
    def _pothole_features_to_render(features: List[dict]) -> List[dict]:
        """Convert pothole feature dicts to RoadRenderer format."""
        out = []
        for f in features:
            out.append({
                "cx": float(np.clip(f["centroid_x"], 0.05, 0.95)),
                "cy": float(np.clip(f["centroid_y"], 0.1, 0.9)),
                "rx": float(np.clip(f["bbox_w"] * 0.5, 0.03, 0.2)),
                "ry": float(np.clip(f["bbox_h"] * 0.5, 0.02, 0.15)),
            })
        return out
