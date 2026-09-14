"""
scripts/demo.py
Quick command-line demo: process one image through the full pipeline
and print results without the Streamlit GUI.

Usage:
    python scripts/demo.py --image path/to/image.jpg
    python scripts/demo.py --demo_name demo_pothole_centre
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from app.pipeline import InferencePipeline, ACTION_NAMES
from app.demo_images import get_demo_image


def main():
    parser = argparse.ArgumentParser(description="CLI demo for pothole avoidance pipeline")
    parser.add_argument("--image",     default=None, help="Path to input image")
    parser.add_argument("--demo_name", default="demo_pothole_centre",
                        help="Use a built-in demo image")
    parser.add_argument("--ckpt",      default=None, help="Perception checkpoint path")
    args = parser.parse_args()

    # Load image
    if args.image:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
    else:
        image_bgr = get_demo_image(args.demo_name)

    # Pipeline
    pipe = InferencePipeline(config_dir=str(ROOT / "configs"))
    if args.ckpt:
        ok = pipe.load_perception_checkpoint(args.ckpt)
        print(f"Checkpoint {'loaded ✓' if ok else 'FAILED — demo mode'}")

    print(f"\n{'='*60}")
    print("POTHOLE AVOIDANCE PIPELINE — CLI DEMO")
    print(f"{'='*60}")
    if pipe.demo_mode:
        print("⚠️  DEMO MODE: no trained checkpoint loaded\n")

    result = pipe.run(image_bgr)

    print(f"Inference latency   : {result.inference_ms:.1f} ms  "
          f"(≈ {1000/max(result.inference_ms,1):.1f} FPS)")
    print(f"Potholes detected   : {result.n_potholes}")
    print(f"State dimension     : {len(result.state_vector)} (must be 117)")
    assert len(result.state_vector) == 117, "STATE DIM ERROR"
    print(f"Max severity        : {result.state_vector[6]:.3f}")
    print(f"Mean uncertainty    : {result.state_vector[0]:.3f}")

    print(f"\n--- RL Agent Probabilities ---")
    print(f"{'Agent':<16} {'Maintain':>8} {'Left':>8} {'Right':>8}")
    print("-" * 42)
    for name, probs in result.agent_probs.items():
        print(f"{name.replace('_',' ').title():<16} {probs[0]:>8.3f} {probs[1]:>8.3f} {probs[2]:>8.3f}")
    ep = result.ensemble_probs
    print("-" * 42)
    print(f"{'Ensemble':<16} {ep[0]:>8.3f} {ep[1]:>8.3f} {ep[2]:>8.3f}")
    print(f"\nRL Decision         : {ACTION_NAMES[result.rl_action]}")

    if result.srl_decision:
        srl = result.srl_decision
        print(f"\n--- SRL ---")
        print(f"SRL Verdict         : {'ACCEPTED ✅' if srl.accepted else 'OVERRIDDEN ⚠️'}")
        if not srl.accepted:
            print(f"Override reason     : {srl.override_reason}")
        print(f"Final Action        : {ACTION_NAMES[result.final_action]}")

    # Save road visualisation
    if result.road_image is not None:
        out_path = ROOT / "data" / "demo_output.png"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), result.road_image)
        print(f"\nRoad simulation saved → {out_path}")

    print(f"\n{'='*60}")
    print("DONE")


if __name__ == "__main__":
    main()
