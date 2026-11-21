"""Stable operational errors for command-line and future UI callers."""


class AnalysisError(Exception):
    code = "analysis_error"


class InvalidImageError(AnalysisError):
    code = "invalid_image"


class ServiceUnavailableError(AnalysisError):
    code = "service_unavailable"


class ModelNotFoundError(AnalysisError):
    code = "model_missing"


class AnalysisTimeoutError(AnalysisError):
    code = "timeout"


class InvalidModelOutputError(AnalysisError):
    code = "invalid_model_output"


class InferenceError(AnalysisError):
    code = "inference_failed"
