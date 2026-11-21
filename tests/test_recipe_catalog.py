from recipe_catalog import load_catalog, recommend_recipes
from fruit_backend.schemas import Assessment


def make_assessment(**overrides: object) -> Assessment:
    values: dict[str, object] = {
        "status": "assessed",
        "fruit_name": "banana",
        "ripeness": "ripe",
        "visible_spoilage": "none_observed",
        "observations": ["Yellow peel"],
        "explanation": "The fruit appears ripe.",
    }
    values.update(overrides)
    if values["status"] != "assessed":
        values["fruit_name"] = None
        values["ripeness"] = "uncertain"
        values["visible_spoilage"] = "uncertain"
    return Assessment.model_validate(values)


def test_catalog_loads_and_is_validated() -> None:
    catalog = load_catalog()
    assert len(catalog) >= 10
    assert {recipe.id for recipe in catalog} == {recipe.id for recipe in catalog}


def test_recommendations_prioritize_exact_ripeness() -> None:
    recommendations = recommend_recipes(make_assessment(ripeness="overripe"))

    assert recommendations
    assert recommendations[0].id == "banana-bread"
    assert all(recipe.fruit == "banana" for recipe in recommendations)


def test_recommendations_include_any_fallback_after_exact_matches() -> None:
    recommendations = recommend_recipes(make_assessment(fruit_name="apple", ripeness="ripe"))

    assert [recipe.id for recipe in recommendations] == ["apple-compote", "apple-slaw"]


def test_recommendations_suppress_clear_spoilage_and_non_assessed_scenes() -> None:
    assert recommend_recipes(make_assessment(visible_spoilage="clear")) == []
    assert recommend_recipes(make_assessment(status="no_fruit", fruit_name=None)) == []


def test_recommendations_respect_limit_and_have_stable_order() -> None:
    first = recommend_recipes(make_assessment(fruit_name="apple", ripeness="uncertain"), limit=1)
    second = recommend_recipes(make_assessment(fruit_name="apple", ripeness="uncertain"), limit=1)

    assert len(first) == 1
    assert first == second
