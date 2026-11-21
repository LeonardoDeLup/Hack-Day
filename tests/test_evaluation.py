import hashlib
import json

import pytest

from fruit_backend.evaluation import load_cases, summarize


def row(fruit="banana", ripeness="ripe", error=None, expected="ripe"):
    return {
        "expected_status": "assessed", "expected_fruit": "banana", "fruit_aliases": [],
        "expected_ripeness": expected, "expected_visible_spoilage": "none_observed",
        "error": error,
        "result": None if error else {
            "assessment": {"status": "assessed", "fruit_name": fruit, "ripeness": ripeness, "visible_spoilage": "none_observed"},
            "metadata": {"inference_seconds": 1},
        },
    }


def test_errors_and_abstentions_do_not_inflate_accuracy():
    summary = summarize([row(), row(ripeness="uncertain"), row(error={"code": "timeout"})])
    assert summary["fruit_identification"] == {"correct": 2, "total": 3, "rate": 0.6667}
    assert summary["ripeness_on_assessable_images"] == {"correct": 1, "total": 3, "rate": 0.3333}
    assert summary["abstentions_on_assessable_ripeness"] == 1
    assert summary["operational_errors"] == 1


def test_correct_ripeness_of_wrong_fruit_is_not_counted():
    summary = summarize([row(fruit="orange")])
    assert summary["ripeness_on_assessable_images"]["correct"] == 0


def test_ambiguous_reference_is_scored_separately():
    summary = summarize([row(expected="uncertain", ripeness="uncertain")])
    assert summary["ripeness_on_assessable_images"]["total"] == 0
    assert summary["uncertainty_on_ambiguous_ripeness"]["correct"] == 1


def fixture_case(image_bytes):
    return {
        "id": "sample", "file": "image.png", "sha256": hashlib.sha256(image_bytes).hexdigest(),
        "split": "development", "source_group": "one-photo", "expected_status": "assessed",
        "expected_fruit": "banana", "fruit_aliases": [], "expected_ripeness": "ripe",
        "expected_visible_spoilage": "none_observed", "label_note": "Fixture", "source": {},
    }


def test_manifest_rejects_changed_image(tmp_path, image_bytes):
    case = fixture_case(image_bytes)
    (tmp_path / "image.png").write_bytes(b"changed image")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([case]))
    with pytest.raises(ValueError, match="checksum"):
        load_cases(manifest)


def test_related_photos_cannot_leak_into_holdout(tmp_path, image_bytes):
    case = fixture_case(image_bytes)
    (tmp_path / "image.png").write_bytes(image_bytes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([case, case | {"id": "derived", "split": "held_out"}]))
    with pytest.raises(ValueError, match="cross"):
        load_cases(manifest)


def test_manifest_rejects_duplicate_case_ids(tmp_path, image_bytes):
    case = fixture_case(image_bytes)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps([case, case]))
    with pytest.raises(ValueError, match="unique"):
        load_cases(manifest)
