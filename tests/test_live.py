"""Optional real-model smoke check, separate from the deterministic suite."""

import os
from pathlib import Path

import pytest

from fruit_backend import analyze_image


@pytest.mark.integration
@pytest.mark.skipif(os.environ.get("RUN_OLLAMA_TESTS") != "1", reason="Set RUN_OLLAMA_TESTS=1 for local-model integration checks")
def test_real_banana_assessment():
    root = Path(__file__).resolve().parents[1]
    result = analyze_image((root / "evaluation/images/banana-yellow.jpg").read_bytes())
    assert result.assessment.status == "assessed"
    assert result.assessment.fruit_name == "banana"
    assert result.assessment.ripeness == "ripe"
