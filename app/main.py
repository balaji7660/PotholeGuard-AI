"""
app/main.py
Main Streamlit application for the Pothole Avoidance Research Prototype.

Run with:
    streamlit run app/main.py

SOFTWARE-ONLY PROTOTYPE — no physical camera required.
Dataset images act as simulated camera input.
"""
from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Make project root importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline import InferencePipeline, ACTION_NAMES, PipelineResult
from app.demo_images import get_demo_image, DEMO_IMAGE_NAMES, list_demo_images

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Pothole Avoidance | Research Prototype",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
  .main { background-color: #0e1117; }
  .title-block {
      background: linear-gradient(135deg, #1a1f2e 0%, #0d1117 100%);
      border: 1px solid #30363d;
      border-radius: 12px;
      padding: 24px 32px;
      margin-bottom: 24px;
  }
  .title-block h1 { color: #58a6ff; font-size: 1.6rem; margin-bottom: 4px; }
  .title-block p  { color: #8b949e; font-size: 0.9rem; }
  .metric-card {
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 8px;
  }
  .metric-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { color: #e6edf3; font-size: 1.1rem; font-weight: 600; }
  .badge-safe     { background:#1f6e3c; color:#3fb950; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
  .badge-override { background:#6e2020; color:#f85149; padding:2px 10px; border-radius:12px; font-size:0.8rem; font-weight:600; }
  .badge-demo     { background:#3d3006; color:#e3b341; padding:2px 10px; border-radius:12px; font-size:0.78rem; font-weight:600; }
  .pipeline-step {
      background:#161b22; border:1px solid #30363d; border-radius:6px;
      padding:8px 14px; margin:4px 0; font-size:0.82rem; color:#c9d1d9;
  }
  .severity-low    { color: #3fb950; font-weight:600; }
  .severity-medium { color: #e3b341; font-weight:600; }
  .severity-high   { color: #f85149; font-weight:600; }
  .stProgress > div > div > div { background: #58a6ff; }
</style>
""", unsafe_allow_html=True)

# ── Cached pipeline ────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline() -> InferencePipeline:
    return InferencePipeline(config_dir=str(ROOT / "configs"))

pipeline = load_pipeline()

# ── Helpers ────────────────────────────────────────────────────────────────────

def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """Apply plasma colormap to depth map, return uint8 RGB."""
    d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    colored = (cm.plasma(d_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored

def colorize_uncertainty(unc: np.ndarray) -> np.ndarray:
    """Apply cool colormap to uncertainty map, return uint8 RGB."""
    u_norm = np.clip(unc, 0, 1)
    colored = (cm.hot(u_norm)[:, :, :3] * 255).astype(np.uint8)
    return colored

def seg_overlay(image_rgb: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Overlay segmentation mask in red on the original image."""
    overlay = image_rgb.copy()
    mask    = (seg >= 0.5)
    overlay[mask] = (overlay[mask] * 0.4 + np.array([220, 40, 40]) * 0.6).astype(np.uint8)
    # Contour
    contours, _ = cv2.findContours(
        (seg >= 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (255, 80, 80), 2)
    return overlay

def prob_bar_fig(agent_probs: dict, ensemble_probs: np.ndarray) -> plt.Figure:
    """Create a compact probability bar chart for all 4 agents + ensemble."""
    labels  = ["Maintain", "Left", "Right"]
    agents  = list(agent_probs.keys()) + ["Ensemble"]
    colors  = ["#58a6ff", "#3fb950", "#f85149", "#e3b341", "#c9d1d9"]
    probs   = [agent_probs[k] for k in agent_probs.keys()] + [ensemble_probs]

    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    x = np.arange(len(labels))
    bar_w = 0.15
    for i, (agent, prob, col) in enumerate(zip(agents, probs, colors)):
        offset = (i - 2) * bar_w
        bars = ax.bar(x + offset, prob, bar_w,
                      label=agent.replace("_", " ").title(),
                      color=col, alpha=0.85, edgecolor="#30363d")
        for bar, val in zip(bars, prob):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color="#c9d1d9", fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Probability", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    legend = ax.legend(fontsize=7, loc="upper right",
                       facecolor="#1c2128", edgecolor="#30363d",
                       labelcolor="#c9d1d9")
    ax.set_title("Agent Action Probability Distributions", color="#58a6ff", fontsize=10, pad=8)
    fig.tight_layout()
    return fig

def ablation_fig() -> plt.Figure:
    """Ablation comparison bar chart (reported reference values)."""
    labels = ["RL-Only\n(Baseline)", "Ensemble RL", "Ensemble RL\n+ SRL"]
    values = [0.3924, 0.3825, 0.2873]
    colors = ["#f85149", "#e3b341", "#3fb950"]

    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    bars = ax.bar(labels, values, color=colors, edgecolor="#30363d", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.4f} m", ha="center", va="bottom", color="#c9d1d9", fontsize=10)

    ax.set_ylabel("Mean Lateral Deviation (m)", color="#8b949e", fontsize=10)
    ax.set_ylim(0, 0.5)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.set_title("Ablation Study — Lateral Deviation\n(Reported Reference Results — NOT locally reproduced)",
                 color="#58a6ff", fontsize=10, pad=8)
    ax.axhline(0.2873, color="#3fb950", linestyle="--", linewidth=1, alpha=0.5)
    fig.tight_layout()
    return fig

def severity_badge(s: float) -> str:
    if s < 0.33:   return f'<span class="severity-low">LOW ({s:.2f})</span>'
    elif s < 0.66: return f'<span class="severity-medium">MEDIUM ({s:.2f})</span>'
    else:          return f'<span class="severity-high">HIGH ({s:.2f})</span>'

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    input_mode = st.radio(
        "Input Source",
        ["📂 Upload Image", "🎬 Upload Video", "🖼️ Demo Images", "📷 Live Camera"],
        index=2,
    )

    uploaded_file = None
    selected_demo = None
    image_bgr     = None

    if input_mode == "📂 Upload Image":
        uploaded_file = st.file_uploader("Upload pothole image",
                                          type=["jpg", "jpeg", "png", "bmp"])
        if uploaded_file:
            file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    elif input_mode == "🎬 Upload Video":
        st.info("Upload a pothole video — frames are processed sequentially.")
        uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])
        st.caption("Video mode: use Run Full Pipeline below after upload.")

    elif input_mode == "📷 Live Camera":
        camera_img = st.camera_input("Take a photo of the road / pothole")
        if camera_img:
            file_bytes = np.frombuffer(camera_img.read(), np.uint8)
            image_bgr  = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.info("💡 For continuous real-time video streaming & HUD on mobile, run: `python run_mobile.py`")

    else:  # Demo Images
        demo_names = list_demo_images()
        selected_demo = st.selectbox("Select demo image", demo_names)
        image_bgr = get_demo_image(selected_demo)
        st.caption("⚠️ Demo images — not from actual benchmark dataset.")

    st.markdown("---")

    # SRL Scenario buttons
    st.markdown("### 🛡️ SRL Scenarios")
    scenario_choice = st.radio(
        "Demo scenario",
        ["None", "A: Safe Left", "B: Left Blocked→Right",
         "C: High Uncertainty Override", "D: Emergency Brake"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### 🔧 Config")
    alpha_ema = st.slider("EMA α (smoothing)", 0.1, 1.0, 0.7, 0.05)
    show_state = st.checkbox("Show 117-D state vector", value=False)

    st.markdown("---")
    run_detection = st.button("🔍 Run Detection", use_container_width=True)
    run_pipeline  = st.button("🚀 Run Full Pipeline", use_container_width=True,
                               type="primary")

    st.markdown("---")
    st.markdown("**Checkpoint**")
    ckpt_path = st.text_input("Checkpoint path (optional)", placeholder="/path/to/best.pt")
    if ckpt_path and st.button("Load Checkpoint"):
        ok = pipeline.load_perception_checkpoint(ckpt_path)
        if ok:
            st.success("✅ Checkpoint loaded — running in model mode")
        else:
            st.error("❌ Failed to load checkpoint — staying in DEMO mode")

# ── Title block ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <h1>🚗 Robust Pothole Avoidance in Autonomous Driving</h1>
  <p>Using Multi-Task Transformer Perception and Ensemble Reinforcement Learning</p>
  <p style="margin-top:8px;">
    <span class="badge-demo">SOFTWARE-ONLY PROTOTYPE</span>
    &nbsp;&nbsp;Dataset-driven simulation — no physical vehicle or camera required
  </p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline flow diagram ───────────────────────────────────────────────────────
with st.expander("📊 Pipeline Architecture", expanded=False):
    cols = st.columns(9)
    steps = [
        ("📷", "RGB Input\n512×512"),
        ("🧠", "TransUNet\nPerception"),
        ("📐", "UASA\nState Extraction"),
        ("🎲", "4 RL Agents\n(PPO/A2C/TRPO/RPPO)"),
        ("🗳️", "Soft Voting\nEnsemble"),
        ("🛡️", "Safety\nRefinement"),
        ("🚗", "Simulated\nVehicle"),
        ("📊", "Decision\nOutput"),
    ]
    for i, (icon, label) in enumerate(steps):
        with cols[i]:
            st.markdown(f"<div style='text-align:center;font-size:1.5rem'>{icon}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#8b949e'>{label}</div>", unsafe_allow_html=True)
        if i < len(steps) - 1:
            with cols[i]:
                pass

# ── Preview image ───────────────────────────────────────────────────────────────
if image_bgr is not None:
    preview_rgb = cv2.cvtColor(cv2.resize(image_bgr, (512, 512)), cv2.COLOR_BGR2RGB)
    col_prev, _ = st.columns([1, 2])
    with col_prev:
        st.markdown("**📸 Input Image**")
        st.image(preview_rgb, caption="Input (resized 512×512)", use_container_width=True)

# ── Main pipeline run ───────────────────────────────────────────────────────────
result: Optional[PipelineResult] = None

if (run_pipeline or run_detection) and image_bgr is not None:
    with st.spinner("Running pipeline..."):
        pipeline.ema.alpha = alpha_ema
        result = pipeline.run(image_bgr)

elif scenario_choice != "None" and image_bgr is not None:
    with st.spinner(f"Running SRL Scenario {scenario_choice[0]}..."):
        result = pipeline.run(image_bgr)
        # Force SRL scenario
        if result and result.srl_decision and pipeline.srl:
            binary_mask = (result.segmentation >= 0.5).astype(np.uint8)
            sc = scenario_choice[0]
            if   sc == "A": result.srl_decision = pipeline.srl.scenario_a(binary_mask)
            elif sc == "B": result.srl_decision = pipeline.srl.scenario_b(binary_mask)
            elif sc == "C": result.srl_decision = pipeline.srl.scenario_c(binary_mask)
            elif sc == "D": result.srl_decision = pipeline.srl.scenario_d(binary_mask)
            if result.srl_decision:
                result.final_action = result.srl_decision.final_action

# ── Results display ─────────────────────────────────────────────────────────────
if result is not None:

    # Demo mode warning
    if result.demo_mode:
        st.warning("⚠️ **DEMO MODE** — Model checkpoint not loaded. "
                   "Perception outputs use classical CV approximations, not a trained TransUNet. "
                   "RL outputs are deterministic mock distributions. Results do NOT represent trained model performance.")

    # ── Row 1: Perception outputs ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔬 Perception Outputs")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Original RGB**")
        st.image(result.original_rgb, use_container_width=True)

    with col2:
        st.markdown("**Pothole Segmentation**")
        seg_vis = seg_overlay(result.original_rgb, result.segmentation)
        st.image(seg_vis, use_container_width=True)
        st.caption(f"Detected regions: {result.n_potholes}")

    with col3:
        st.markdown("**Depth Map** *(pseudo-depth)*")
        depth_vis = colorize_depth(result.depth)
        st.image(depth_vis, use_container_width=True)
        st.caption("Color: blue=near, yellow=far (demo approximation)")

    with col4:
        st.markdown("**Uncertainty Map**")
        unc_vis = colorize_uncertainty(result.uncertainty)
        st.image(unc_vis, use_container_width=True)
        mean_unc = float(result.uncertainty.mean())
        unc_label = "LOW" if mean_unc < 0.3 else ("MEDIUM" if mean_unc < 0.6 else "HIGH")
        st.caption(f"Mean uncertainty: {mean_unc:.3f} ({unc_label})")

    # ── Row 2: Features + State ─────────────────────────────────────────────
    st.markdown("---")
    col_feat, col_state = st.columns([1, 2])

    with col_feat:
        st.markdown("### 📐 Pothole Features")
        if result.n_potholes == 0:
            st.info("No potholes detected in this frame.")
        else:
            for i, f in enumerate(result.pothole_features):
                with st.expander(f"Pothole {i+1}  |  Severity: {f['severity']:.2f}", expanded=(i==0)):
                    cols_f = st.columns(2)
                    metrics = [
                        ("Centroid X", f"{f['centroid_x']:.3f}"),
                        ("Centroid Y", f"{f['centroid_y']:.3f}"),
                        ("BBox Width", f"{f['bbox_w']:.3f}"),
                        ("BBox Height", f"{f['bbox_h']:.3f}"),
                        ("Area (px)", f"{f['area']:.0f}"),
                        ("Lane Disp. (m)", f"{f['lane_displacement']:.3f}"),
                        ("Avg Depth", f"{f['avg_depth']:.3f}"),
                        ("Min Depth", f"{f['min_depth']:.3f}"),
                        ("Depth Var.", f"{f['depth_var']:.4f}"),
                        ("Uncertainty", f"{f['uncertainty']:.3f}"),
                    ]
                    for j, (lbl, val) in enumerate(metrics):
                        with cols_f[j % 2]:
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-label">{lbl}</div>'
                                f'<div class="metric-value">{val}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    st.markdown(
                        f'Severity: {severity_badge(f["severity"])}',
                        unsafe_allow_html=True
                    )

    with col_state:
        st.markdown("### 🔢 117-D UASA Risk-Aware State Vector")
        state = result.state_vector
        assert len(state) == 117, f"State dimension error: {len(state)}"

        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("State Dimension", "117 ✓")
            st.metric("Frame Uncertainty", f"{state[0]:.3f}")
            st.metric("Pothole Coverage", f"{state[2]:.3f}")
        with col_s2:
            st.metric("Num Potholes (norm)", f"{state[3]:.2f}")
            st.metric("Max Severity", f"{state[6]:.3f}")
            st.metric("Mean Severity", f"{state[7]:.3f}")
        with col_s3:
            st.metric("Min Depth (norm)", f"{state[8]:.3f}")
            st.metric("Global UncMax", f"{state[1]:.3f}")
            st.metric("EMA-smoothed", "✓")

        # State vector heatmap
        fig_state, ax_state = plt.subplots(figsize=(10, 1.5))
        fig_state.patch.set_facecolor("#0d1117")
        ax_state.set_facecolor("#161b22")
        im = ax_state.imshow(
            state.reshape(1, -1), aspect="auto", cmap="RdYlGn_r",
            vmin=0, vmax=1
        )
        ax_state.set_yticks([])
        ax_state.set_xlabel("Feature Index (0–116)", color="#8b949e", fontsize=8)
        ax_state.tick_params(colors="#8b949e", labelsize=7)
        for spine in ax_state.spines.values():
            spine.set_edgecolor("#30363d")
        plt.colorbar(im, ax=ax_state, orientation="vertical", pad=0.01, shrink=0.8)
        ax_state.set_title("117-D State Vector (heatmap)", color="#58a6ff", fontsize=9)
        fig_state.tight_layout()
        st.pyplot(fig_state)
        plt.close(fig_state)

        if show_state:
            st.code(np.array2string(state, precision=4, separator=", ",
                                    max_line_width=120), language="text")

    # ── Row 3: RL Ensemble ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🎲 RL Ensemble — Soft Voting")

    if result.demo_mode:
        st.caption("⚠️ Demo-mode RL: mock distributions based on UASA features. Not a trained policy.")

    # Probability table
    agents  = list(result.agent_probs.keys())
    col_rl1, col_rl2 = st.columns([3, 2])

    with col_rl1:
        fig_probs = prob_bar_fig(result.agent_probs, result.ensemble_probs)
        st.pyplot(fig_probs)
        plt.close(fig_probs)

    with col_rl2:
        st.markdown("**Action Probability Table**")
        table_rows = []
        for name, probs in result.agent_probs.items():
            table_rows.append({
                "Agent": name.replace("_", " ").title(),
                "Maintain": f"{probs[0]:.3f}",
                "Left": f"{probs[1]:.3f}",
                "Right": f"{probs[2]:.3f}",
            })
        ep = result.ensemble_probs
        table_rows.append({
            "Agent": "**Ensemble**",
            "Maintain": f"**{ep[0]:.3f}**",
            "Left": f"**{ep[1]:.3f}**",
            "Right": f"**{ep[2]:.3f}**",
        })
        import pandas as pd
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

        rl_label = ACTION_NAMES[result.rl_action]
        st.markdown(
            f'<div style="margin-top:12px;padding:12px;background:#1f2937;border-radius:8px;">'
            f'<div style="color:#8b949e;font-size:0.75rem">RL DECISION (argmax ensemble)</div>'
            f'<div style="color:#58a6ff;font-size:1.4rem;font-weight:700">{rl_label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Row 4: SRL ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛡️ Safety Refinement Layer")

    if result.srl_decision:
        srl = result.srl_decision
        col_srl1, col_srl2, col_srl3 = st.columns(3)

        with col_srl1:
            st.markdown(f"**RL Proposed:** {srl.rl_label}")
            st.markdown(f"**SRL Verdict:**")
            if srl.accepted:
                st.markdown('<span class="badge-safe">✅ ACCEPTED</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-override">⚠️ OVERRIDDEN</span>', unsafe_allow_html=True)
                if srl.override_reason:
                    st.error(f"Reason: {srl.override_reason}")

        with col_srl2:
            st.markdown("**Safety Checks:**")
            checks = {
                "Uncertainty": srl.uncertainty_ok,
                "Severity":    srl.severity_ok,
                "Deviation":   srl.deviation_ok,
                "Lane":        srl.lane_ok,
                "Consistency": srl.consistency_ok,
            }
            for name, ok in checks.items():
                icon = "✅" if ok else "❌"
                color = "#3fb950" if ok else "#f85149"
                st.markdown(f'<span style="color:{color}">{icon} {name}</span>',
                            unsafe_allow_html=True)

        with col_srl3:
            st.markdown("**Collision Map:**")
            for action_id, collides in srl.collision_map.items():
                icon = "💥" if collides else "✅"
                st.markdown(
                    f'{icon} Action {action_id} ({ACTION_NAMES[action_id]}): '
                    f'{"COLLISION" if collides else "Safe"}',
                )

        final_label = ACTION_NAMES.get(result.final_action, str(result.final_action))
        if srl.override_reason and "EMERGENCY" in srl.override_reason:
            final_label = "🚨 EMERGENCY BRAKE"

        st.markdown(
            f'<div style="margin-top:12px;padding:16px;background:#1f2937;border-radius:8px;border:1px solid #30363d;">'
            f'<div style="color:#8b949e;font-size:0.8rem;margin-bottom:4px">FINAL ACTION</div>'
            f'<div style="color:#3fb950;font-size:1.6rem;font-weight:700">{final_label}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── Row 5: Simulation ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛣️ Simulated Road & Vehicle Trajectory")

    if result.road_image is not None:
        road_rgb = cv2.cvtColor(result.road_image, cv2.COLOR_BGR2RGB)
        col_road, col_road_info = st.columns([3, 1])
        with col_road:
            st.image(road_rgb, caption="Top-down road simulation", use_container_width=True)
        with col_road_info:
            st.markdown("**Vehicle State**")
            st.metric("Lateral Offset", f"{pipeline.vehicle.x_norm:.3f}")
            st.metric("Lateral Dev. (m)", f"{pipeline.vehicle.lateral_deviation_m:.3f} m")
            st.markdown("**Trajectory Legend**")
            st.markdown("🟢 Safe trajectory")
            st.markdown("🔴 Unsafe trajectory")
            st.markdown("🟡 Alternative action")
            st.markdown("🟠 SRL override")
            st.markdown("**FPS**")
            fps = 1000.0 / max(result.inference_ms, 1)
            st.metric("Demo FPS", f"{fps:.1f}")
            st.caption("(CPU demo — GPU would be faster)")

    # ── FPS & timing ────────────────────────────────────────────────────────
    st.markdown(f"⏱️ Pipeline latency: **{result.inference_ms:.1f} ms** "
                f"(≈ **{1000/max(result.inference_ms,1):.1f} FPS** on this hardware in demo mode)")

# ── Tabs: Dashboard + Ablation ─────────────────────────────────────────────────
st.markdown("---")
tab1, tab2, tab3 = st.tabs(["📊 Results Dashboard", "🔬 Ablation Study", "ℹ️ About"])

with tab1:
    st.markdown("### 📊 Reference Results from Research Report")
    st.info("⚠️ These values are **reported results from the research paper**, "
            "not reproduced by this local prototype.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dice (CV mean)", "0.8393 ± 0.0148", delta="Reported")
    c2.metric("Final Dice",     "0.8356",            delta="Reported")
    c3.metric("IoU",            "0.7344",            delta="Reported")
    c4.metric("Depth RMSE",     "4.1570",            delta="Reported")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Full-system Lateral Dev.", "0.289 m",  delta="Reported")
    c6.metric("RL-only Baseline",         "0.392 m",  delta="Reported")
    c7.metric("Improvement",              "26.3%",    delta="Reported")
    c8.metric("SRL Intervention Rate",    "55.9%",    delta="Reported")

    st.metric("Reported Inference Speed", "≈ 22 FPS",
              help="Reported for 16-GB GPU; local demo FPS depends on hardware.")

with tab2:
    st.markdown("### 🔬 Ablation Study")
    st.info("Reference values from the research report — clearly labeled as reported, not locally reproduced.")
    fig_abl = ablation_fig()
    st.pyplot(fig_abl)
    plt.close(fig_abl)

    import pandas as pd
    abl_df = pd.DataFrame({
        "Configuration":         ["A. RL-Only", "B. Ensemble RL", "C. Ensemble RL + SRL"],
        "Lateral Deviation (m)": [0.3924,       0.3825,           0.2873],
        "Improvement vs A":      ["–",          "2.5%",           "26.8%"],
        "Source":                ["Reported",   "Reported",       "Reported"],
    })
    st.dataframe(abl_df, hide_index=True, use_container_width=True)

with tab3:
    st.markdown("""
    ### About This Application

    **This is a software-only research prototype** implementing:
    > "Robust Pothole Avoidance in Autonomous Driving Using Multi-Task Transformer
    > Perception and Ensemble Reinforcement Learning"

    #### Demo Mode
    When no trained checkpoint is loaded, the system uses:
    - Classical CV (Otsu thresholding) as a stand-in for TransUNet
    - Deterministic mock RL distributions derived from UASA state features
    - All outputs are clearly labeled as **DEMO**

    #### Key Components
    | Component | Description |
    |---|---|
    | TransUNet | Multi-task CNN + Transformer + U-Net decoder |
    | UASA | 117-dimensional risk-aware state vector |
    | PPO / A2C / TRPO / RecPPO | 4 independent RL agents |
    | Soft Voting | Equal-weight ensemble (implementation choice) |
    | SRL | Deterministic 7-check safety gate |
    | Simulation | Top-down road visualisation |

    #### Research Integrity Note
    - Reference benchmark values are **displayed only**, not claimed to be reproduced here.
    - All reward coefficients, severity weights, and ensemble weights are **implementation choices**.
    - This system does **not** control a physical vehicle.
    """)

# ── Video mode ─────────────────────────────────────────────────────────────────
if input_mode == "🎬 Upload Video" and uploaded_file is not None and run_pipeline:
    st.markdown("---")
    st.markdown("### 🎬 Video Processing")
    tfile = io.BytesIO(uploaded_file.read())
    # Save temporarily
    tmp_path = ROOT / "data" / "tmp_video.mp4"
    tmp_path.parent.mkdir(exist_ok=True)
    with open(tmp_path, "wb") as f:
        f.write(tfile.read())

    cap = cv2.VideoCapture(str(tmp_path))
    frame_placeholder = st.empty()
    info_placeholder  = st.empty()
    pipeline.vehicle.reset()

    frame_idx = 0
    fps_list  = []
    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break
        t0 = time.perf_counter()
        res = pipeline.run(frame_bgr)
        fps_list.append(1.0 / max(time.perf_counter() - t0, 0.001))

        # Display every 3rd frame for speed
        if frame_idx % 3 == 0:
            road_rgb = cv2.cvtColor(res.road_image, cv2.COLOR_BGR2RGB) if res.road_image is not None else None
            col_v1, col_v2 = st.columns(2)
            seg_v = seg_overlay(res.original_rgb, res.segmentation)
            frame_placeholder.image(
                np.hstack([res.original_rgb, seg_v]),
                caption=f"Frame {frame_idx} | Action: {ACTION_NAMES[res.final_action]}",
                use_container_width=True,
            )
            if road_rgb is not None:
                info_placeholder.image(road_rgb, caption="Road simulation", use_container_width=True)

        frame_idx += 1
        if frame_idx > 300:   # Safety limit
            break

    cap.release()
    avg_fps = np.mean(fps_list) if fps_list else 0.0
    st.success(f"✅ Processed {frame_idx} frames | Demo avg: {avg_fps:.1f} FPS")
