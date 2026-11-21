"""Deterministic, offline recipe recommendations for visually assessed fruit.

Catalog: ``recipes/catalog.json`` (relative to this module, not the process cwd).
The model's fruit label is normalized conservatively, with explicit aliases.
"""

from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fruit_backend.schemas import Assessment

Ripeness = Literal["unripe", "ripe", "overripe", "uncertain", "any"]


class Recipe(BaseModel):
    # Keep the original schema so streamlit_app.py remains compatible.
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


# Explicit synonyms only: do not guess a species from a generic name like "citrus".
FRUIT_ALIASES: dict[str, str] = {
    "apples": "apple", "green-apple": "apple", "red-apple": "apple",
    "granny-smith-apple": "apple", "honeycrisp-apple": "apple",
    "bananas": "banana", "cavendish-banana": "banana",
    "green-banana": "banana", "yellow-banana": "banana",
    "plantains": "plantain", "cooking-banana": "plantain",
    "green-plantain": "plantain", "yellow-plantain": "plantain",
    "pears": "pear", "asian-pear": "pear", "nashi": "pear",
    "peaches": "peach", "nectarines": "nectarine", "plums": "plum",
    "apricots": "apricot", "cherries": "cherry", "sweet-cherry": "cherry",
    "strawberries": "strawberry", "raspberries": "raspberry",
    "blackberries": "blackberry", "blueberries": "blueberry",
    "cranberries": "cranberry", "mulberries": "mulberry",
    "gooseberries": "gooseberry", "black-currant": "blackcurrant",
    "red-currant": "redcurrant", "black-currants": "blackcurrant",
    "red-currants": "redcurrant", "grapes": "grape",
    "oranges": "orange", "navel-orange": "orange", "blood-orange": "orange",
    "sweet-orange": "orange", "mandarin-orange": "mandarin",
    "mandarin-oranges": "mandarin", "tangerine": "mandarin",
    "tangerines": "mandarin", "clementine": "mandarin",
    "clementines": "mandarin", "satsuma": "mandarin",
    "satsumas": "mandarin", "grapefruits": "grapefruit",
    "lemons": "lemon", "limes": "lime", "pomelos": "pomelo",
    "pummelo": "pomelo", "kumquats": "kumquat",
    "mangos": "mango", "mangoes": "mango", "green-mango": "mango",
    "papayas": "papaya", "green-papaya": "papaya",
    "pawpaw-papaya": "papaya", "pineapples": "pineapple",
    "kiwifruit": "kiwi", "kiwi-fruit": "kiwi", "kiwis": "kiwi",
    "guavas": "guava", "passionfruit": "passion-fruit",
    "passion-fruits": "passion-fruit", "pitaya": "dragon-fruit",
    "pitahaya": "dragon-fruit", "dragonfruit": "dragon-fruit",
    "dragon-fruits": "dragon-fruit", "pomegranates": "pomegranate",
    "arils": "pomegranate", "water-melon": "watermelon",
    "watermelons": "watermelon", "rockmelon": "cantaloupe",
    "muskmelon": "cantaloupe", "cantaloupes": "cantaloupe",
    "honeydew-melon": "honeydew", "honeydew-melons": "honeydew",
    "persimmons": "persimmon", "figs": "fig", "lychees": "lychee",
    "litchi": "lychee", "litchis": "lychee", "rambutans": "rambutan",
    "feijoas": "feijoa", "pineapple-guava": "feijoa",
    "sapodillas": "sapodilla", "chikoo": "sapodilla",
    "cherimoyas": "cherimoya", "custard-apple-cherimoya": "cherimoya",
    "prickly-pears": "prickly-pear", "cactus-pear": "prickly-pear",
    "soursops": "soursop", "guanabana": "soursop",
    "avocados": "avocado", "avocado-pear": "avocado",
    "tomatoes": "tomato", "green-tomato": "tomato", "green-tomatoes": "tomato",
    "dates": "date", "medjool-date": "date", "coconuts": "coconut",
    "young-coconut": "coconut", "jackfruits": "jackfruit",
    "breadfruits": "breadfruit", "tamarinds": "tamarind",
    "durians": "durian", "black-sapotes": "black-sapote",
    "chocolate-pudding-fruit": "black-sapote",
    "quinces": "quince", "loquats": "loquat",
}


@lru_cache(maxsize=4)
def load_catalog(path: str | Path | None = None) -> tuple[Recipe, ...]:
    """Load and strictly validate the bundled recipe JSON once per process.

    Use ``load_catalog.cache_clear()`` when editing the JSON during development.
    """
    catalog_path = Path(path) if path is not None else Path(__file__).resolve().parent / "recipes" / "catalog.json"
    raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Recipe catalog must contain a JSON list.")
    result = tuple(Recipe.model_validate(item) for item in raw)
    ids = [recipe.id for recipe in result]
    if len(ids) != len(set(ids)):
        raise ValueError("Recipe catalog contains duplicate recipe IDs.")
    return result


def normalize_fruit_name(value: str) -> str:
    """Normalize known alternatives without guessing an unspecified fruit variety."""
    name = _slug(value)
    known = {_slug(recipe.fruit) for recipe in load_catalog()}
    if name in known:
        return name
    if name in FRUIT_ALIASES:
        return FRUIT_ALIASES[name]
    # Generic plurals not already covered by explicit aliases.
    if name.endswith("s") and name[:-1] in known:
        return name[:-1]
    return name


def recommend_recipes(assessment: Assessment, limit: int = 3) -> list[Recipe]:
    """Return stable fruit/ripeness matches; never invent a recipe.

    When visual spoilage is possible, clear, or uncertain, withhold recipes. An
    image alone cannot establish food safety even when spoilage isn't visible.
    """
    if limit <= 0 or assessment.status != "assessed" or not assessment.fruit_name:
        return []
    if assessment.visible_spoilage != "none_observed":
        return []

    fruit = normalize_fruit_name(assessment.fruit_name)
    ripeness = assessment.ripeness
    scored: list[tuple[int, Recipe]] = []
    for recipe in load_catalog():
        if _slug(recipe.fruit) != fruit:
            continue
        if ripeness in recipe.ripeness:
            scored.append((2, recipe))
        elif "any" in recipe.ripeness:
            scored.append((1, recipe))

    scored.sort(key=lambda item: (-item[0], item[1].id))
    return [recipe for _, recipe in scored[:limit]]
