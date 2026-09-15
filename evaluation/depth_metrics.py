"""
Depth Estimation Evaluation Metrics for PotholeGuard-AI.
Calculates RMSE, MAE, Abs Relative Difference, and Delta Threshold Accuracy (delta < 1.25).
"""
import numpy as np

def compute_depth_metrics(pred_depth: np.ndarray, gt_depth: np.ndarray, mask: np.ndarray = None) -> dict:
    if mask is not None and np.sum(mask > 0) > 0:
        pred = pred_depth[mask > 0].astype(np.float64)
        gt = gt_depth[mask > 0].astype(np.float64)
    else:
        pred = pred_depth.flatten().astype(np.float64)
        gt = gt_depth.flatten().astype(np.float64)

    # Filter invalid/zero gt
    valid = gt > 1e-4
    if np.sum(valid) == 0:
        return {"rmse": 0.0, "mae": 0.0, "abs_rel": 0.0, "delta_1": 1.0}

    p = pred[valid]
    g = gt[valid]

    diff = p - g
    rmse = np.sqrt(np.mean(diff ** 2))
    mae = np.mean(np.abs(diff))
    abs_rel = np.mean(np.abs(diff) / g)

    # Delta threshold metric: max(p/g, g/p) < 1.25
    ratio = np.maximum(p / (g + 1e-6), g / (p + 1e-6))
    delta_1 = np.mean((ratio < 1.25).astype(np.float64))

    return {
        "rmse": float(round(rmse, 4)),
        "mae": float(round(mae, 4)),
        "abs_rel": float(round(abs_rel, 4)),
        "delta_1": float(round(delta_1, 4))
    }
