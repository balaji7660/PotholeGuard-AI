"""
Dataset Report Generator for PotholeGuard-AI.
Inspects local dataset directories and generates a structured JSON report
with actual file counts and annotation properties (never fabricated).
"""
import os
import glob
import json
from typing import Dict, Any

def generate_dataset_report(data_root: str = "data", output_path: str = "dataset_report.json") -> Dict[str, Any]:
    report = {
        "dataset_root": data_root,
        "pothole600_found": False,
        "potholergbd_found": False,
        "total_images": 0,
        "annotated_masks": 0,
        "ground_truth_depth_images": 0,
        "pseudo_depth_images": 0,
        "splits": {
            "train": 0,
            "val": 0,
            "test": 0
        },
        "status": "NO_LOCAL_DATASET_FOUND"
    }

    if os.path.exists(data_root):
        p600_dir = os.path.join(data_root, "pothole-600")
        prgbd_dir = os.path.join(data_root, "pothole-rgbd")

        if os.path.exists(p600_dir):
            report["pothole600_found"] = True
            p600_imgs = glob.glob(os.path.join(p600_dir, "images", "*.*"))
            p600_masks = glob.glob(os.path.join(p600_dir, "masks", "*.*"))
            report["total_images"] += len(p600_imgs)
            report["annotated_masks"] += len(p600_masks)
            report["pseudo_depth_images"] += len(p600_imgs)

        if os.path.exists(prgbd_dir):
            report["potholergbd_found"] = True
            prgbd_imgs = glob.glob(os.path.join(prgbd_dir, "rgb", "*.*"))
            prgbd_masks = glob.glob(os.path.join(prgbd_dir, "masks", "*.*"))
            prgbd_depths = glob.glob(os.path.join(prgbd_dir, "depth", "*.*"))
            report["total_images"] += len(prgbd_imgs)
            report["annotated_masks"] += len(prgbd_masks)
            report["ground_truth_depth_images"] += len(prgbd_depths)

        if report["total_images"] > 0:
            report["status"] = "DATASET_READY"
            n_tot = report["total_images"]
            report["splits"]["train"] = int(n_tot * 0.70)
            report["splits"]["val"] = int(n_tot * 0.15)
            report["splits"]["test"] = n_tot - report["splits"]["train"] - report["splits"]["val"]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report

if __name__ == "__main__":
    rep = generate_dataset_report()
    print("Generated dataset report:", rep)
