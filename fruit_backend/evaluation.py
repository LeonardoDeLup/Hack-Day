"""Small, reproducible visual-label evaluation; no food-safety ground truth."""

import argparse
import hashlib
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from .analysis import GENERATION_OPTIONS, FruitAnalyzer
from .errors import AnalysisError
from .prompts import PROMPT_VERSION
from .schemas import Ripeness, Spoilage, Status


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    file: str
    sha256: str
    split: Literal["development", "held_out"]
    source_group: str
    expected_status: Status
    expected_fruit: str | None
    fruit_aliases: list[str]
    expected_ripeness: Ripeness
    expected_visible_spoilage: Spoilage
    label_note: str
    source: dict[str, Any]


def load_cases(manifest: Path) -> list[EvaluationCase]:
    cases = TypeAdapter(list[EvaluationCase]).validate_json(manifest.read_text())
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("The evaluation needs nonempty, unique case IDs.")
    groups: dict[str, str] = {}
    for case in cases:
        if case.source_group in groups and groups[case.source_group] != case.split:
            raise ValueError("Related source images must not cross development/held-out splits.")
        groups[case.source_group] = case.split
        image_path = (manifest.parent / case.file).resolve()
        if not image_path.is_relative_to(manifest.parent.resolve()):
            raise ValueError(f"Image path escapes the evaluation directory: {case.id}")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != case.sha256:
            raise ValueError(f"Image checksum does not match the labeled manifest: {case.id}")
    return cases


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count errors/abstentions as misses on assessable references, never omit them."""
    def assessment(row: dict[str, Any]) -> dict[str, Any]:
        return cast(dict[str, Any], (row.get("result") or {}).get("assessment", {}))

    def fraction(correct: int, total: int) -> dict[str, int | float | None]:
        return {"correct": correct, "total": total, "rate": round(correct / total, 4) if total else None}

    fruit_rows = [r for r in records if r["expected_status"] == "assessed"]
    ripeness_rows = [r for r in fruit_rows if r["expected_ripeness"] != "uncertain"]
    uncertain_rows = [r for r in fruit_rows if r["expected_ripeness"] == "uncertain"]
    spoilage_rows = [r for r in fruit_rows if r["expected_visible_spoilage"] != "uncertain"]
    rejection_rows = [r for r in records if r["expected_status"] != "assessed"]
    times = [r["result"]["metadata"]["inference_seconds"] for r in records if r.get("result")]
    def fruit_correct(row: dict[str, Any]) -> bool:
        prediction = assessment(row)
        return prediction.get("status") == "assessed" and prediction.get("fruit_name") in [row["expected_fruit"], *row["fruit_aliases"]]

    return {
        "cases": len(records),
        "operational_errors": sum(r.get("error") is not None for r in records),
        "invalid_outputs": sum((r.get("error") or {}).get("code") == "invalid_model_output" for r in records),
        "status": fraction(sum(assessment(r).get("status") == r["expected_status"] for r in records), len(records)),
        "fruit_identification": fraction(sum(fruit_correct(r) for r in fruit_rows), len(fruit_rows)),
        "ripeness_on_assessable_images": fraction(sum(fruit_correct(r) and assessment(r).get("ripeness") == r["expected_ripeness"] for r in ripeness_rows), len(ripeness_rows)),
        "abstentions_on_assessable_ripeness": sum(assessment(r).get("ripeness") == "uncertain" for r in ripeness_rows),
        "uncertainty_on_ambiguous_ripeness": fraction(sum(fruit_correct(r) and assessment(r).get("ripeness") == "uncertain" for r in uncertain_rows), len(uncertain_rows)),
        "visible_spoilage": fraction(sum(fruit_correct(r) and assessment(r).get("visible_spoilage") == r["expected_visible_spoilage"] for r in spoilage_rows), len(spoilage_rows)),
        "scene_rejection": fraction(sum(assessment(r).get("status") == r["expected_status"] for r in rejection_rows), len(rejection_rows)),
        "first_inference_seconds": times[0] if times else None,
        "subsequent_median_seconds": round(statistics.median(times[1:]), 4) if len(times) > 1 else None,
        "max_inference_seconds": max(times) if times else None,
    }


def run_evaluation(
    manifest: Path,
    output: Path,
    split: Literal["development", "held_out", "all"] = "development",
    overwrite: bool = False,
) -> dict[str, Any]:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Report already exists: {output}. Choose a new path or pass --overwrite.")
    cases = load_cases(manifest)
    selected = [case for case in cases if split == "all" or case.split == split]
    if not selected:
        raise ValueError(f"No cases in split {split}.")
    analyzer = FruitAnalyzer()
    digest = analyzer.check_model()
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "prompt_version": PROMPT_VERSION, "model_digest": digest,
        "generation_options": {**GENERATION_OPTIONS, "think": True},
        "split": split, "complete": False, "records": records,
        "label_basis": "Implementation-assistant visual review before inference; provisional appearance labels, not expert, laboratory maturity, or food-safety ground truth.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)

    for index, case in enumerate(selected, 1):
        record = case.model_dump()
        try:
            result = analyzer.analyze_image((manifest.parent / case.file).read_bytes())
            record.update(result=result.model_dump(), error=None)
            prediction = result.assessment
            print(f"[{index}/{len(selected)}] {case.id}: {prediction.status}, {prediction.fruit_name}, {prediction.ripeness} ({result.metadata.inference_seconds:.1f}s)", file=sys.stderr, flush=True)
        except AnalysisError as exc:
            record.update(result=None, error={"code": exc.code, "message": str(exc)})
            print(f"[{index}/{len(selected)}] {case.id}: {exc.code}", file=sys.stderr, flush=True)
        records.append(record)
        report["summary"] = summarize(records)
        report["complete"] = index == len(selected)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate local Qwen3-VL against frozen visual labels.")
    parser.add_argument("--manifest", type=Path, default=Path("evaluation/manifest.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=["development", "held_out", "all"], default="development")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = run_evaluation(args.manifest, args.output, args.split, args.overwrite)
    except (AnalysisError, OSError, ValueError, ValidationError) as exc:
        print(json.dumps({"error": getattr(exc, "code", "evaluation_error"), "message": str(exc)}), file=sys.stderr)
        return 1
    print(json.dumps(report["summary"], indent=2))
    return 1 if report["summary"]["operational_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
