"""
Segmentation Metrics for PotholeGuard-AI.
Calculates: Dice score, Intersection over Union (IoU), Precision, Recall, Specificity.
"""
import numpy as np

def compute_segmentation_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict:
    pred_b = (pred_mask > 0).astype(np.uint8)
    gt_b = (gt_mask > 0).astype(np.uint8)

    tp = np.sum((pred_b == 1) & (gt_b == 1))
    fp = np.sum((pred_b == 1) & (gt_b == 0))
    fn = np.sum((pred_b == 0) & (gt_b == 1))
    tn = np.sum((pred_b == 0) & (gt_b == 0))

    smooth = 1e-6
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    accuracy = (tp + tn + smooth) / (tp + tn + fp + fn + smooth)

    return {
        "dice": float(round(dice, 4)),
        "iou": float(round(iou, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "accuracy": float(round(accuracy, 4)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }
