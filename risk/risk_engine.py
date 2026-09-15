"""
Comprehensive Risk Intelligence Engine for PotholeGuard-AI.
Evaluates 9 dynamic hazard factors:
1. Size score (pixel area & dimensions)
2. Relative depth score (depression depth)
3. Shape irregularity factor (circularity, elongation, boundary complexity)
4. Surface severity score
5. Rider-path relevance (proximity to centerline corridor)
6. Approach rate (growth & longitudinal proximity)
7. Temporal tracking stability (persistence)
8. Perception uncertainty (epistemic/aleatoric variance)
9. Optical image quality risk

Outputs Risk Level:
- SAFE (0.00 - 0.29)
- WARNING (0.30 - 0.59)
- HIGH_WARNING (0.60 - 0.79)
- DANGER (0.80 - 1.00)
- UNCERTAIN (Triggered on elevated uncertainty / optical degradation)

Includes Explainable AI (XAI) rationale for warnings.
"""
from typing import Dict, Any, Optional
from risk.thresholds import RISK_TIER_THRESHOLDS
from risk.risk_explainer import RiskExplainer

class RiskEngine:
    def __init__(self, config: Optional[Dict[str, float]] = None):
        self.config = config or {
            "w_size": 0.20,
            "w_depth": 0.25,
            "w_shape": 0.10,
            "w_severity": 0.15,
            "w_path": 0.15,
            "w_approach": 0.10,
            "w_temporal": 0.05,
            "w_uncertainty": 0.00,  # acts as overriding factor
            "w_quality": 0.00,
            "safe_max": RISK_TIER_THRESHOLDS["SAFE_MAX"],
            "warning_max": RISK_TIER_THRESHOLDS["WARNING_MAX"],
            "high_warning_max": RISK_TIER_THRESHOLDS["HIGH_WARNING_MAX"],
            "danger_min": RISK_TIER_THRESHOLDS["DANGER_MIN"],
            "uncertainty_threshold": RISK_TIER_THRESHOLDS["UNCERTAINTY_TRIGGER"]
        }
        self.explainer = RiskExplainer()

    def update_config(self, new_config: Dict[str, float]):
        self.config.update(new_config)

    def evaluate_risk(self, features: Dict[str, Any]) -> Dict[str, Any]:
        size_score = float(features.get("size_score", 0.0))
        depth_score = float(features.get("depth_score", 0.0))
        shape_factor = float(features.get("shape_severity_factor", 0.5))
        severity_score = float(features.get("severity_score", 0.0))
        path_score = float(features.get("path_relevance_score", 0.0))
        approach_score = float(features.get("approach_score", 0.3))
        temporal_stability = float(features.get("temporal_stability", 0.8))
        uncertainty = float(features.get("uncertainty", 0.0))
        quality_score = float(features.get("quality_score", 1.0))

        # Check uncertainty override first
        if uncertainty >= self.config["uncertainty_threshold"] or quality_score < 0.35:
            risk_class = "UNCERTAIN"
            recommendation = "CAUTION - UNCERTAIN DETECTION"
            risk_score = round(
                self.config["w_size"] * size_score +
                self.config["w_depth"] * depth_score +
                self.config["w_shape"] * shape_factor +
                self.config["w_severity"] * severity_score +
                self.config["w_path"] * path_score +
                self.config["w_approach"] * approach_score,
                3
            )
            reasons = self.explainer.explain_risk(features, risk_class)
            return {
                "risk_score": risk_score,
                "risk_class": risk_class,
                "recommendation": recommendation,
                "is_urgent": False,
                "uncertainty_alert": True,
                "reasons": reasons
            }

        # Multi-factor scalar score calculation
        risk_score = round(
            self.config["w_size"] * size_score +
            self.config["w_depth"] * depth_score +
            self.config["w_shape"] * shape_factor +
            self.config["w_severity"] * severity_score +
            self.config["w_path"] * path_score +
            self.config["w_approach"] * approach_score +
            self.config["w_temporal"] * (1.0 - temporal_stability),
            3
        )

        position_class = features.get("position_class", "CENTER / DIRECT PATH")

        if risk_score >= self.config["danger_min"]:
            risk_class = "DANGER"
            if position_class == "CENTER / DIRECT PATH":
                recommendation = "SLOW DOWN - DEEP HAZARD AHEAD"
            elif position_class == "LEFT OF PATH":
                recommendation = "HAZARD ON LEFT - STAY RIGHT"
            else:
                recommendation = "HAZARD ON RIGHT - STAY LEFT"
            is_urgent = True
        elif risk_score > self.config["warning_max"]:
            risk_class = "HIGH_WARNING"
            recommendation = "HIGH CAUTION - POTHOLE APPROACHING"
            is_urgent = True
        elif risk_score > self.config["safe_max"]:
            risk_class = "WARNING"
            recommendation = "CAUTION - POTHOLE AHEAD"
            is_urgent = False
        else:
            risk_class = "SAFE"
            recommendation = "MAINTAIN COURSE"
            is_urgent = False

        reasons = self.explainer.explain_risk(features, risk_class)

        return {
            "risk_score": risk_score,
            "risk_class": risk_class,
            "recommendation": recommendation,
            "is_urgent": is_urgent,
            "uncertainty_alert": False,
            "reasons": reasons
        }
