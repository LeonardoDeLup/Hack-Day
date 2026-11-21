"""Local-only inference boundary. No Streamlit, persistence, retries, or global client."""

import hashlib
import json
import time
from typing import Callable, TypeVar

import httpx
import ollama
from pydantic import ValidationError

from .errors import (
    AnalysisTimeoutError, InferenceError, InvalidModelOutputError,
    ModelNotFoundError, ServiceUnavailableError,
)
from .images import prepare_image
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, USER_PROMPT
from .schemas import AnalysisMetadata, AnalysisResult, Assessment, response_json_schema

MODEL = "qwen3-vl:4b"
HOST = "http://127.0.0.1:11434"
# Thinking is explicit and separate from the final JSON in message.content.
# 768 tokens truncated JSON; reserve room for both thinking and the final response.
GENERATION_OPTIONS: dict[str, int | float] = {
    "temperature": 0.1, "top_p": 0.95, "top_k": 20,
    "repeat_penalty": 1.0, "presence_penalty": 0.0, "seed": 1234,
    "num_ctx": 4096, "num_predict": 1024,
}
T = TypeVar("T")


def _request(call: Callable[[], T]) -> T:
    try:
        return call()
    except httpx.TimeoutException as exc:
        raise AnalysisTimeoutError("Ollama timed out. Retry explicitly or check available memory and CPU/GPU load.") from exc
    except (ConnectionError, httpx.ConnectError) as exc:
        raise ServiceUnavailableError(f"Cannot reach Ollama at {HOST}. Start it with 'ollama serve'.") from exc
    except ollama.ResponseError as exc:
        if exc.status_code == 404:
            raise ModelNotFoundError(f"Model {MODEL} is unavailable. Run 'ollama pull {MODEL}'.") from exc
        raise InferenceError(f"Ollama returned an error (HTTP {exc.status_code}): {exc.error}") from exc
    except httpx.HTTPError as exc:
        raise InferenceError("The connection to Ollama failed during the request.") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise InferenceError("Ollama returned an invalid response envelope.") from exc


class FruitAnalyzer:
    """Reuse an instance for batch evaluation; each image still gets a fresh chat."""

    def __init__(self, client: ollama.Client | None = None) -> None:
        self._client = client if client is not None else ollama.Client(
            host=HOST, timeout=httpx.Timeout(180.0, connect=3.0), trust_env=False,
        )
        self._model_digest: str | None = None

    def check_model(self) -> str:
        """Check availability and return the installed model digest."""
        response = _request(self._client.list)
        for model in response.models:
            if model.model == MODEL:
                self._model_digest = model.digest or "unavailable"
                return self._model_digest
        raise ModelNotFoundError(f"Model {MODEL} is not installed. Run 'ollama pull {MODEL}'.")

    def analyze_image(self, image_bytes: bytes) -> AnalysisResult:
        started = time.perf_counter()
        print("1. Preparing image...", flush=True)
        prepared = prepare_image(image_bytes)
        print(f"2. Image prepared: {prepared.width}x{prepared.height}", flush=True)
        if self._model_digest is None:
            print("3. Checking installed model...", flush=True)
            self.check_model()
        schema = response_json_schema()
        user_prompt = USER_PROMPT
        inference_started = time.perf_counter()
        print("4. Starting Ollama inference...", flush=True)
        response = _request(lambda: self._client.chat(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt, "images": [prepared.data]},
            ],
            format=schema,
            stream=False,
            think=False,
            keep_alive="5m",
            options=GENERATION_OPTIONS,
        ))
        inference_seconds = time.perf_counter() - inference_started
        print("5. Ollama inference completed!", flush=True)
        if response.done_reason == "length":
            raise InvalidModelOutputError(
                f"The model reached its generation limit ({response.eval_count} tokens) before finishing the assessment. Retry explicitly."
            )
        if not response.message.content:
            raise InvalidModelOutputError("The model returned an empty final assessment. Retry explicitly.")
        try:
            assessment = Assessment.model_validate_json(response.message.content)
        except ValidationError as exc:
            raise InvalidModelOutputError("The model response failed assessment validation. Retry explicitly.") from exc

        return AnalysisResult(
            assessment=assessment,
            metadata=AnalysisMetadata(
                model=MODEL, model_digest=self._model_digest or "unavailable",
                prompt_version=PROMPT_VERSION,
                prompt_sha256=hashlib.sha256((SYSTEM_PROMPT + "\n" + user_prompt + "\n" + json.dumps(schema, sort_keys=True)).encode()).hexdigest(),
                image_sha256=prepared.original_sha256,
                prepared_image_sha256=prepared.prepared_sha256,
                prepared_width=prepared.width, prepared_height=prepared.height,
                elapsed_seconds=round(time.perf_counter() - started, 4),
                inference_seconds=round(inference_seconds, 4),
                model_load_seconds=(response.load_duration / 1_000_000_000 if response.load_duration is not None else None),
                output_tokens=response.eval_count,
                generation_options={**GENERATION_OPTIONS, "think": True},
                thinking_output_present=bool(response.message.thinking),
            ),
        )


def analyze_image(image_bytes: bytes) -> AnalysisResult:
    """Analyze one image using the local qwen3-vl:4b model."""
    return FruitAnalyzer().analyze_image(image_bytes)
