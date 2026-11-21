"""Deterministic, local recipe recommendations for assessed fruit."""

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fruit_backend.schemas import Assessment

Ripeness = Literal["unripe", "ripe", "overripe", "uncertain", "any"]


class Recipe(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    id: str
    title: str
    fruit: str
    ripeness: list[Ripeness] = Field(min_length=1)
    summary: str
    why_it_fits: str
    prep_minutes: int = Field(ge=0)
    cook_minutes: int = Field(ge=0)
    servings: int = Field(ge=1)
    ingredients: list[str] = Field(min_length=1)
    steps: list[str] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


@lru_cache(maxsize=1)
def load_catalog(path: str = "recipes/catalog.json") -> tuple[Recipe, ...]:
    """Load and validate the bundled catalog once per process."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Recipe catalog must contain a JSON list.")
    return tuple(Recipe.model_validate(item) for item in raw)


def recommend_recipes(assessment: Assessment, limit: int = 3) -> list[Recipe]:
    """Return stable fruit/ripeness matches; never invent a recipe."""
    if assessment.status != "assessed" or assessment.fruit_name is None:
        return []
    if assessment.visible_spoilage == "clear":
        return []

    fruit = _slug(assessment.fruit_name)
    ripeness = assessment.ripeness
    scored: list[tuple[int, Recipe]] = []
    for recipe in load_catalog():
        if _slug(recipe.fruit) != fruit:
            continue
        recipe_ripeness = {_slug(value) for value in recipe.ripeness}
        if ripeness in recipe_ripeness:
            scored.append((2, recipe))
        elif "any" in recipe_ripeness:
            scored.append((1, recipe))
    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [recipe for _, recipe in scored[:limit]]
