"""
detection/model.py
Pothole detection and perception model loader.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent.resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline import InferencePipeline

def load_model():
    """Load and return the perception & decision pipeline model."""
    pipeline = InferencePipeline(config_dir=str(ROOT / "configs"))
    return pipeline
