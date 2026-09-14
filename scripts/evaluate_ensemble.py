"""
scripts/evaluate_ensemble.py
Evaluate the full ensemble pipeline on test images.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from omegaconf import OmegaConf
from app.pipeline import InferencePipeline
from app.demo_images import get_demo_image, DEMO_IMAGE_NAMES
from evaluation.metrics import PipelineMetrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default=None, help="Directory of test images")
    parser.add_argument("--ckpt",      default=None)
    args = parser.parse_args()

    pipe    = InferencePipeline(config_dir=str(ROOT / "configs"))
    metrics = PipelineMetrics()

    if args.image_dir:
        image_dir = Path(args.image_dir)
        paths     = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    else:
        # Use demo images
        paths = None
        print("No image_dir provided — using demo images.")

    images = []
    if paths:
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append(img)
    else:
        for name in DEMO_IMAGE_NAMES:
            images.append(get_demo_image(name))

    print(f"Evaluating on {len(images)} images...")
    for i, img in enumerate(images):
        result = pipe.run(img)
        srl_override = (result.srl_decision is not None and not result.srl_decision.accepted)
        metrics.add(
            lateral_deviation=pipe.vehicle.lateral_deviation_m,
            srl_override=srl_override,
            fps=1000.0 / max(result.inference_ms, 1),
            action=result.final_action,
        )
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(images)} processed")

    summary = metrics.summary()
    print("\n=== Evaluation Summary (Demo Mode) ===")
    for k, v in summary.items():
        print(f"  {k:<25}: {v:.4f}")
    print("\nNote: These are DEMO results, not trained-model benchmark results.")


if __name__ == "__main__":
    main()
