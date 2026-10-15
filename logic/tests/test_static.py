"""Static checks: algorithm modules never read the wall clock."""
import re
from pathlib import Path

import pytest

LOGIC = Path(__file__).resolve().parents[1]
ALGORITHM_MODULES = ["filters", "features", "fall", "gait", "tug", "risk"]
FORBIDDEN = re.compile(r"time\.time\(|datetime\.now\(|time\.monotonic\(|time\.perf_counter\(|^\s*import time\b|"
                       r"^\s*from time import|^\s*from datetime import", re.MULTILINE)


@pytest.mark.parametrize("module", ALGORITHM_MODULES)
def test_no_wall_clock_in_algorithm_code(module) -> None:
    path = LOGIC / f"{module}.py"
    if not path.exists():
        pytest.skip(f"{module}.py not written yet")
    assert not FORBIDDEN.search(path.read_text(encoding="utf-8")), f"wall clock used in {module}.py"
