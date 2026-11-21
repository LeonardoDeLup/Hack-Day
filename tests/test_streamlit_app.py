from unittest.mock import patch
from pathlib import Path

from streamlit.testing.v1 import AppTest

from fruit_backend.errors import ServiceUnavailableError
from fruit_backend.schemas import AnalysisMetadata, AnalysisResult, Assessment


def assessed_result() -> AnalysisResult:
    assessment = Assessment(
        status="assessed",
        fruit_name="banana",
        ripeness="overripe",
        visible_spoilage="none_observed",
        observations=["Darkened peel", "Soft-looking fruit"],
        explanation="The banana looks very ripe and suitable for baking.",
    )
    metadata = AnalysisMetadata(
        model="qwen3-vl:4b",
        model_digest="test",
        prompt_version="test",
        prompt_sha256="test",
        image_sha256="test",
        prepared_image_sha256="test",
        prepared_width=1,
        prepared_height=1,
        elapsed_seconds=0.1,
        inference_seconds=None,
        model_load_seconds=None,
        output_tokens=None,
        generation_options={"think": True},
        thinking_output_present=True,
    )
    return AnalysisResult(assessment=assessment, metadata=metadata)


def uploaded_app() -> AppTest:
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run()
    image_bytes = (Path(__file__).parents[1] / "evaluation/images/banana-dark.jpg").read_bytes()
    app.file_uploader("fruit_upload").upload("banana.jpg", image_bytes)
    return app.run()


def test_app_renders_initial_upload_flow() -> None:
    app = AppTest.from_file(Path(__file__).parents[1] / "streamlit_app.py").run()

    assert not app.exception
    assert app.file_uploader[0].label == "Upload a fruit photo"
    assert len(app.button) == 0


def test_app_renders_assessment_and_matching_recipe() -> None:
    app = uploaded_app()
    with patch("ui_services.analyze_uploaded_image", return_value=assessed_result()):
        app.button(key="analyze_button").click().run()

    assert not app.exception
    assert any("Banana" in item.value for item in app.markdown)
    assert any("One-bowl banana bread" in item.value for item in app.markdown)


def test_app_surfaces_local_model_error() -> None:
    app = uploaded_app()
    error = ServiceUnavailableError("Ollama is unavailable")
    with patch("ui_services.analyze_uploaded_image", side_effect=error):
        app.button(key="analyze_button").click().run()

    assert not app.exception
    assert any("local model is offline" in item.value.lower() for item in app.error)
