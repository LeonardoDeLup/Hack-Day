import json
from unittest.mock import Mock

import httpx
import ollama
import pytest
from pydantic import ValidationError

from fruit_backend import (
    AnalysisTimeoutError, FruitAnalyzer, InferenceError, InvalidImageError,
    InvalidModelOutputError, ModelNotFoundError, ServiceUnavailableError,
)
from fruit_backend.analysis import MODEL
from fruit_backend.schemas import Assessment, response_json_schema


def client_for(assessment):
    client = Mock(spec=ollama.Client)
    client.list.return_value = ollama.ListResponse(models=[{"model": MODEL, "digest": "test-digest"}])
    client.chat.return_value = ollama.ChatResponse(
        model=MODEL, message={"role": "assistant", "content": json.dumps(assessment)},
        done=True, done_reason="stop", load_duration=100000000, eval_count=42,
    )
    return client


def test_request_and_metadata(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    result = FruitAnalyzer(client).analyze_image(image_bytes)
    assert result.assessment.fruit_name == "banana"
    assert result.metadata.model_digest == "test-digest"
    assert result.metadata.model_load_seconds == 0.1
    kwargs = client.chat.call_args.kwargs
    assert kwargs["model"] == MODEL
    assert kwargs["format"] == response_json_schema()
    assert kwargs["stream"] is False
    assert kwargs["think"] is True
    assert kwargs["options"]["num_predict"] == 4096
    assert kwargs["options"]["temperature"] == 0.6
    assert kwargs["options"]["seed"] == 1234
    assert result.metadata.generation_options["num_predict"] == 4096
    assert result.metadata.generation_options["think"] is True
    assert isinstance(kwargs["messages"][1]["images"][0], bytes)
    assert all("image_sha256" not in branch["properties"] for branch in kwargs["format"]["anyOf"])


def test_batch_checks_model_once_and_uses_independent_chats(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    analyzer = FruitAnalyzer(client)
    analyzer.analyze_image(image_bytes)
    analyzer.analyze_image(image_bytes)
    assert client.list.call_count == 1
    assert client.chat.call_count == 2
    assert all(len(call.kwargs["messages"]) == 2 for call in client.chat.call_args_list)


def test_thinking_is_separate_from_validated_output(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    client.chat.return_value.message.thinking = "internal fixture text, not JSON"
    result = FruitAnalyzer(client).analyze_image(image_bytes)
    assert result.assessment.fruit_name == "banana"
    assert result.metadata.thinking_output_present is True
    assert "internal fixture text" not in result.model_dump_json()


def test_invalid_image_never_contacts_ollama(valid_assessment):
    client = client_for(valid_assessment)
    with pytest.raises(InvalidImageError):
        FruitAnalyzer(client).analyze_image(b"broken")
    client.list.assert_not_called()
    client.chat.assert_not_called()


@pytest.mark.parametrize("error,expected", [
    (ConnectionError("offline"), ServiceUnavailableError),
    (httpx.ConnectError("offline"), ServiceUnavailableError),
    (httpx.ReadTimeout("slow"), AnalysisTimeoutError),
    (httpx.ConnectTimeout("slow"), AnalysisTimeoutError),
    (ollama.ResponseError("missing", 404), ModelNotFoundError),
    (ollama.ResponseError("out of memory", 500), InferenceError),
    (httpx.ReadError("reset"), InferenceError),
    (json.JSONDecodeError("bad response", "", 0), InferenceError),
])
def test_typed_failures_without_retries(image_bytes, valid_assessment, error, expected):
    client = client_for(valid_assessment)
    client.chat.side_effect = error
    with pytest.raises(expected):
        FruitAnalyzer(client).analyze_image(image_bytes)
    assert client.chat.call_count == 1


def test_missing_model(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    client.list.return_value = ollama.ListResponse(models=[{"model": "some-other-model"}])
    with pytest.raises(ModelNotFoundError):
        FruitAnalyzer(client).analyze_image(image_bytes)
    client.chat.assert_not_called()


def test_preflight_connection_failure(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    client.list.side_effect = ConnectionError("offline")
    with pytest.raises(ServiceUnavailableError):
        FruitAnalyzer(client).analyze_image(image_bytes)
    client.chat.assert_not_called()


@pytest.mark.parametrize("content", ["", "not JSON", "```json\n{}\n```", '{"status":"assessed"}'])
def test_bad_output(image_bytes, valid_assessment, content):
    client = client_for(valid_assessment)
    client.chat.return_value.message.content = content
    with pytest.raises(InvalidModelOutputError):
        FruitAnalyzer(client).analyze_image(image_bytes)


def test_truncation_is_not_accepted_even_with_valid_json(image_bytes, valid_assessment):
    client = client_for(valid_assessment)
    client.chat.return_value.done_reason = "length"
    with pytest.raises(InvalidModelOutputError):
        FruitAnalyzer(client).analyze_image(image_bytes)


@pytest.mark.parametrize("changes", [
    {"fruit_name": None}, {"fruit_name": "unknown"}, {"status": "no_fruit"},
    {"ripeness": "edible"}, {"confidence": 0.99}, {"observations": []},
])
def test_schema_rejects_inconsistent_or_invented_fields(valid_assessment, changes):
    with pytest.raises(ValidationError):
        Assessment.model_validate(valid_assessment | changes)


def test_uncertainty_is_valid_for_identified_fruit(valid_assessment):
    assert Assessment.model_validate(valid_assessment | {"ripeness": "uncertain"}).status == "assessed"


@pytest.mark.parametrize("status", ["no_fruit", "multiple_fruits", "unclear_image"])
def test_rejected_scene_has_no_condition_claims(valid_assessment, status):
    result = Assessment.model_validate(valid_assessment | {
        "status": status, "fruit_name": None, "ripeness": "uncertain", "visible_spoilage": "uncertain",
    })
    assert result.fruit_name is None
