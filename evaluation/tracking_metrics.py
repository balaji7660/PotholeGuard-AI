"""
Tracking and Warning Evaluation Metrics for PotholeGuard-AI.
Evaluates ID switches, track continuity, duplicate warning suppression rate.
"""
from typing import List, Dict

def compute_tracking_metrics(
    total_ground_truth_potholes: int,
    total_assigned_tracks: int,
    id_switches: int,
    total_warning_events: int,
    duplicate_warnings_blocked: int
) -> Dict:
    duplicate_rate = (
        (total_assigned_tracks - total_ground_truth_potholes) / max(1, total_ground_truth_potholes)
    )
    suppression_efficiency = (
        duplicate_warnings_blocked / max(1, total_warning_events + duplicate_warnings_blocked)
    )

    return {
        "ground_truth_potholes": total_ground_truth_potholes,
        "assigned_tracks": total_assigned_tracks,
        "id_switches": id_switches,
        "duplicate_detection_rate": max(0.0, round(duplicate_rate, 4)),
        "total_warnings_issued": total_warning_events,
        "duplicate_warnings_blocked": duplicate_warnings_blocked,
        "suppression_efficiency": round(suppression_efficiency, 4)
    }
