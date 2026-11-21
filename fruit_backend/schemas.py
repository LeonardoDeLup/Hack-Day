"""Model output and application-owned metadata are deliberately separate."""

from copy import deepcopy
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Status = Literal["assessed", "no_fruit", "multiple_fruits", "unclear_image"]
Ripeness = Literal["unripe", "ripe", "overripe", "uncertain"]
Spoilage = Literal["none_observed", "possible", "clear", "uncertain"]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=180)]
FruitName = Annotated[str, StringConstraints(strip_whitespace=True, to_lower=True, min_length=1, max_length=60)]


class Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Status
    fruit_name: FruitName | None
    ripeness: Ripeness
    visible_spoilage: Spoilage
    observations: list[ShortText] = Field(min_length=1, max_length=4)
    explanation: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]

    @model_validator(mode="after")
    def consistent_assessment(self) -> Self:
        if self.status == "assessed":
            if self.fruit_name is None or self.fruit_name in {"unknown", "uncertain", "fruit", "none", "n/a"}:
                raise ValueError("An assessed result must identify a fruit.")
        elif self.fruit_name is not None or self.ripeness != "uncertain" or self.visible_spoilage != "uncertain":
            raise ValueError("Non-assessed results require null fruit and uncertain condition fields.")
        return self


class AnalysisMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    model_digest: str
    prompt_version: str
    prompt_sha256: str
    image_sha256: str
    prepared_image_sha256: str
    prepared_width: int
    prepared_height: int
    elapsed_seconds: float
    inference_seconds: float
    model_load_seconds: float | None
    output_tokens: int | None
    generation_options: dict[str, int | float | bool]
    thinking_output_present: bool


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessment: Assessment
    metadata: AnalysisMetadata


def response_json_schema() -> dict[str, Any]:
    """Constrain cross-field consistency during generation as well as parsing.

    Use two complete top-level alternatives: llama.cpp does not support mixing
    properties with anyOf at the same level, nor if/then conditionals.
    """
    assessed = deepcopy(Assessment.model_json_schema())
    assessed["properties"]["status"] = {"type": "string", "enum": ["assessed"]}
    assessed["properties"]["fruit_name"] = {"type": "string", "minLength": 1, "maxLength": 60}
    rejected = deepcopy(Assessment.model_json_schema())
    rejected["properties"]["status"] = {
        "type": "string", "enum": ["no_fruit", "multiple_fruits", "unclear_image"],
    }
    rejected["properties"]["fruit_name"] = {"type": "null"}
    for field in ("ripeness", "visible_spoilage"):
        rejected["properties"][field] = {"type": "string", "enum": ["uncertain"]}
    return {"title": "FruitAssessment", "anyOf": [assessed, rejected]}
