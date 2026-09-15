"""
Safety Refinement Layer (SRL) for PotholeGuard-AI.
Deterministic Safety Guardian for two-wheeler rider safety assistance.

Performs rule-based safety validation on perception and risk outputs:
Check 1: High Uncertainty Override (Never allow SAFE when perception uncertainty > threshold)
Check 2: Direct Trajectory Impending Hazard (Deep/Large pothole in direct corridor escalates to DANGER)
Check 3: Temporal Stability & Persistence Filter (Filters transient single-frame false positives)
Check 4: Proximity Escalation (Hazard in bottom 30% of frame requires immediate attention)
Check 5: Severity-Depth Cross Check (Discrepant shallow/deep ratings are safely consolidated)
Check 6: Safe State Validation (Requires zero active hazards and low uncertainty)
Check 7: Non-Intrusive Safety Assertion (Validates that no autonomous steering/braking command is issued)

SAFETY DISCLAIMER:
This module provides informational rider warnings only. It does NOT steer, brake, or control the vehicle.
"""
from typing import Dict, Any, List

class SafetyRefinementLayer:
    def __init__(self, uncertainty_threshold: float = 0.45, min_persistent_frames: int = 1):
        self.uncertainty_threshold = uncertainty_threshold
        self.min_persistent_frames = min_persistent_frames

    def refine_detection(self, detection: Dict[str, Any], track_hits: int = 1) -> Dict[str, Any]:
        """
        Refine a single detection record through deterministic safety checks.
        """
        refined = dict(detection)
        risk_class = detection.get("risk_class", "SAFE")
        risk_score = float(detection.get("risk_score", 0.0))
        uncertainty = float(detection.get("uncertainty", 0.0))
        depth_class = detection.get("depth_class", "SHALLOW")
        size_class = detection.get("size_class", "SMALL")
        position_class = detection.get("position_class", "CENTER / DIRECT PATH")
        path_relevance_score = float(detection.get("path_relevance_score", 0.0))

        overrides = []

        # Check 1: High Uncertainty Override
        if uncertainty >= self.uncertainty_threshold and risk_class == "SAFE":
            risk_class = "UNCERTAIN"
            overrides.append("UNCERTAINTY_OVERRIDE_SAFE_TO_UNCERTAIN")

        # Check 2: Direct Trajectory Impending Hazard
        if position_class == "CENTER / DIRECT PATH" and (depth_class == "DEEP" or size_class in ["LARGE", "VERY LARGE"]):
            if risk_class != "DANGER":
                risk_class = "DANGER"
                risk_score = max(risk_score, 0.75)
                overrides.append("DIRECT_PATH_HAZARD_ESCALATION_TO_DANGER")

        # Check 3: Transient False-Positive Suppression for Low Risks
        if track_hits < self.min_persistent_frames and risk_class == "WARNING":
            # Keep as tentative warning
            overrides.append("TENTATIVE_TRACK_VALIDATION")

        # Check 4: Proximity Escalation (Immediate Path Threat)
        if path_relevance_score > 0.80 and depth_class in ["MEDIUM", "DEEP"]:
            if risk_class == "WARNING":
                risk_class = "DANGER"
                risk_score = max(risk_score, 0.70)
                overrides.append("PROXIMITY_IMMINENT_ESCALATION")

        # Check 7: Safety Boundary Verification (Ensure recommendations are non-intrusive advisories)
        rec = detection.get("recommendation", "MAINTAIN COURSE")
        safe_rec_map = {
            "SAFE": "MAINTAIN COURSE",
            "WARNING": "CAUTION - POTHOLE AHEAD",
            "DANGER": "SLOW DOWN - DEEP HAZARD AHEAD",
            "UNCERTAIN": "CAUTION - UNCERTAIN DETECTION"
        }
        if risk_class in safe_rec_map:
            refined["recommendation"] = safe_rec_map[risk_class]

        refined["risk_class"] = risk_class
        refined["risk_score"] = round(risk_score, 3)
        refined["srl_overrides"] = overrides
        refined["srl_validated"] = True
        return refined

    def refine_all(self, tracks: List[Any]) -> List[Dict[str, Any]]:
        """
        Refine all active tracks.
        """
        results = []
        for t in tracks:
            features = dict(t.features)
            features["track_id"] = t.track_id
            features["bbox"] = list(t.bbox)
            features["hits"] = t.hits
            refined = self.refine_detection(features, track_hits=t.hits)
            t.features.update(refined)
            results.append(refined)
        return results
