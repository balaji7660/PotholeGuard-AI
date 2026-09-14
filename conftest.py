# conftest.py — makes project root importable for pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
