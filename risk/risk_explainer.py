"""
Explainable AI Risk Reasoner for PotholeGuard-AI.
Generates human-interpretable justifications explaining WHY a specific hazard tier was assigned.
"""
from typing import Dict, Any, List

class RiskExplainer:
    def explain_risk(self, features: Dict[str, Any], risk_class: str) -> List[str]:
        reasons = []
        
        # 1. Size
        size_class = features.get("size_class", "MEDIUM")
        if size_class in ["LARGE", "VERY LARGE"]:
            reasons.append(f"Large surface footprint ({size_class})")

        # 2. Depth
        depth_class = features.get("depth_class", "MEDIUM")
        if depth_class in ["DEEP", "VERY DEEP"]:
            reasons.append(f"Significant depression depth ({depth_class})")

        # 3. Path
        position_class = features.get("position_class", "CENTER / DIRECT PATH")
        path_relevance = features.get("relevance_class", "MEDIUM")
        if position_class == "CENTER / DIRECT PATH" or path_relevance == "HIGH":
            reasons.append("Directly in projected motorcycle path")
        elif position_class == "LEFT OF PATH":
            reasons.append("Off-center hazard on left flank")
        elif position_class == "RIGHT OF PATH":
            reasons.append("Off-center hazard on right flank")

        # 4. Approach
        approach_state = features.get("approach_state", "FAR")
        if approach_state == "NEAR":
            reasons.append("Immediate proximity (Critical front-wheel zone)")
        elif approach_state == "APPROACHING":
            reasons.append("Rapidly approaching trajectory")

        # 5. Severity & Shape
        severity_score = float(features.get("severity_score", 0.0))
        if severity_score > 0.65:
            reasons.append("High morphological surface roughness")

        # 6. Uncertainty & Quality
        uncertainty = float(features.get("uncertainty", 0.0))
        quality_tier = features.get("quality_tier", "HIGH")
        if uncertainty >= 0.45:
            reasons.append("Perception uncertainty elevated (Adverse visual condition)")
        if quality_tier == "LOW":
            reasons.append("Optical degradation (Motion blur / Glare detected)")

        if not reasons and risk_class == "SAFE":
            reasons.append("Nominal road surface with clear trajectory")

        return reasons
