"""CLI: JSON assessment on stdout, structured operational errors on stderr."""

import argparse
import json
import sys
from pathlib import Path

from .analysis import analyze_image
from .errors import AnalysisError
from .images import MAX_BYTES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Identify one fruit and assess visible ripeness using local Ollama.")
    parser.add_argument("image", type=Path, help="Path to a JPEG, PNG, or WebP image")
    args = parser.parse_args(argv)
    try:
        with args.image.open("rb") as image_file:
            image_bytes = image_file.read(MAX_BYTES + 1)
        result = analyze_image(image_bytes)
    except OSError as exc:
        print(json.dumps({"error": "file_error", "message": str(exc)}), file=sys.stderr)
        return 2
    except AnalysisError as exc:
        print(json.dumps({"error": exc.code, "message": str(exc)}), file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0
