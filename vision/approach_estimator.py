"""
Approach State and Proximity Estimator for PotholeGuard-AI.
Monitors temporal expansion of bounding boxes and vertical frame descent:
- Categorizes approach status into: FAR, APPROACHING, NEAR
- Computes approach urgency score in [0, 1] without fabricating uncalibrated metric speed.
"""
from typing import Dict, Any, List, Tuple

class ApproachEstimator:
    def estimate_approach(
        self,
        current_bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int],
        track_history: List[Tuple[int, int, int, int]] = None
    ) -> Dict[str, Any]:
        H, W = frame_shape
        x1, y1, x2, y2 = current_bbox
        bot_y = y2
        area_current = max(1, (x2 - x1) * (y2 - y1))

        # Normalized vertical frame position (bottom = closest to motorcycle front wheel)
        proximity_ratio = float(bot_y / float(H))

        # Growth rate across history
        growth_rate = 1.0
        if track_history and len(track_history) >= 2:
            prev_bbox = track_history[0]
            prev_area = max(1, (prev_bbox[2] - prev_bbox[0]) * (prev_bbox[3] - prev_bbox[1]))
            growth_rate = float(area_current / prev_area)

        # Classify Approach State and Range
        if proximity_ratio > 0.72:
            approach_state = "NEAR"
            range_category = "CLOSE RANGE"
            approach_score = 0.95
        elif proximity_ratio > 0.45 or growth_rate > 1.25:
            approach_state = "APPROACHING"
            range_category = "MID RANGE"
            approach_score = 0.65
        else:
            approach_state = "FAR"
            range_category = "LONG RANGE"
            approach_score = 0.30

        return {
            "approach_state": approach_state,
            "range_category": range_category,
            "approach_score": round(approach_score, 3),
            "proximity_ratio": round(proximity_ratio, 3),
            "scale_growth_rate": round(growth_rate, 2),
            "is_imminent": approach_state == "NEAR"
        }
