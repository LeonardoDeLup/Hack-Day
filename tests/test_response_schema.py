import jsonschema
import pytest

from fruit_backend.schemas import response_json_schema


def test_generation_schema_accepts_assessment(valid_assessment):
    jsonschema.validate(valid_assessment, response_json_schema())


@pytest.mark.parametrize("status", ["no_fruit", "multiple_fruits", "unclear_image"])
def test_generation_schema_constrains_rejections(valid_assessment, status):
    schema = response_json_schema()
    valid = valid_assessment | {
        "status": status, "fruit_name": None, "ripeness": "uncertain", "visible_spoilage": "uncertain",
    }
    jsonschema.validate(valid, schema)
    for contradiction in [{"fruit_name": "banana"}, {"ripeness": "ripe"}, {"visible_spoilage": "clear"}]:
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(valid | contradiction, schema)


def test_generation_schema_rejects_assessed_without_fruit(valid_assessment):
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(valid_assessment | {"fruit_name": None}, response_json_schema())
