"""Small UI-facing service boundary, kept free of Streamlit imports."""

from fruit_backend.analysis import FruitAnalyzer
from fruit_backend.schemas import AnalysisResult


def analyze_uploaded_image(image_bytes: bytes, analyzer: FruitAnalyzer) -> AnalysisResult:
    return analyzer.analyze_image(image_bytes)
