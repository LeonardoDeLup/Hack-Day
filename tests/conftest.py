import io

import pytest
from PIL import Image


@pytest.fixture
def image_bytes():
    output = io.BytesIO()
    Image.new("RGB", (80, 40), "yellow").save(output, format="PNG")
    return output.getvalue()


@pytest.fixture
def valid_assessment():
    return {
        "status": "assessed", "fruit_name": "banana", "ripeness": "ripe",
        "visible_spoilage": "none_observed", "observations": ["Yellow peel"],
        "explanation": "The peel is yellow with no visible spoilage.",
    }
