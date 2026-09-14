"""
scripts/evaluate_srl.py
Evaluate SRL intervention rate and scenario responses.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from omegaconf import OmegaConf
from safety.srl import SafetyRefinementLayer

srl_cfg = OmegaConf.load(ROOT / "configs" / "srl_config.yaml")
srl     = SafetyRefinementLayer(srl_cfg)

print("=== SRL Scenario Tests ===\n")
mask = np.zeros((512, 512), dtype=np.uint8)

for name, fn in [
    ("A (Left accepted)", srl.scenario_a),
    ("B (Left→Right override)", srl.scenario_b),
    ("C (High uncertainty)", srl.scenario_c),
    ("D (Emergency brake)", srl.scenario_d),
]:
    srl.reset()
    result = fn(mask)
    status = "ACCEPTED ✅" if result.accepted else f"OVERRIDDEN ⚠️  ({result.override_reason})"
    print(f"Scenario {name}")
    print(f"  RL action   : {result.rl_label}")
    print(f"  SRL verdict : {status}")
    print(f"  Final action: {result.label}\n")

print("=== Done ===")
