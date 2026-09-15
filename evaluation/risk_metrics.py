"""
Risk Assessment Evaluation Metrics for PotholeGuard-AI.
Calculates multi-class confusion matrix, precision, recall, and F1-score across:
SAFE, WARNING, DANGER, UNCERTAIN.
"""
import numpy as np
from typing import List, Dict

CLASSES = ["SAFE", "WARNING", "DANGER", "UNCERTAIN"]

def compute_risk_metrics(y_true: List[str], y_pred: List[str]) -> Dict:
    n_classes = len(CLASSES)
    class_to_idx = {c: i for i, c in enumerate(CLASSES)}
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for yt, yp in zip(y_true, y_pred):
        if yt in class_to_idx and yp in class_to_idx:
            cm[class_to_idx[yt]][class_to_idx[yp]] += 1

    total = max(1, len(y_true))
    correct = np.trace(cm)
    accuracy = correct / float(total)

    per_class = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        prec = (tp / float(tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / float(tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": int(np.sum(cm[i, :]))
        }

    return {
        "overall_accuracy": round(accuracy, 4),
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
        "per_class_metrics": per_class
    }
