"""
Risk and Hazard Classification Thresholds for PotholeGuard-AI.
Configurable scalar tiers for two-wheeler rider safety alerting.
"""
RISK_TIER_THRESHOLDS = {
    "SAFE_MAX": 0.29,
    "WARNING_MAX": 0.59,
    "HIGH_WARNING_MAX": 0.79,
    "DANGER_MIN": 0.80,
    "UNCERTAINTY_TRIGGER": 0.45,
    "POOR_QUALITY_TRIGGER": 0.40
}

RISK_LEVELS = ["SAFE", "WARNING", "HIGH_WARNING", "DANGER", "UNCERTAIN"]
