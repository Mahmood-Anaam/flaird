"""Allow offline source-tree tests without installing training/demo dependencies."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
