# Local fruit assessment backend

Identify one fruit in a photo and estimate its **visible ripeness** using `qwen3-vl:4b` on local Ollama. This phase contains a reusable Python package, a JSON CLI, deterministic tests, and a reproducible evaluation notebook. There is no frontend or recipe integration yet.

## Setup

Tested on this Apple Silicon Mac with Python 3.12 and Ollama 0.40.17. Run commands from the project directory.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
```

For the exact development dependency snapshot, install `requirements-dev.lock` first, then install this package with `--no-deps`:

```sh
.venv/bin/python -m pip install -r requirements-dev.lock
.venv/bin/python -m pip install -e . --no-deps
```

Start Ollama in a separate terminal if it is not already running:

```sh
ollama serve
```

Check the model, and download it only if missing:

```sh
ollama list
ollama pull qwen3-vl:4b
```

The backend connects only to `http://127.0.0.1:11434`, bypasses proxy environment variables for this local connection, and never falls back to a cloud service or another model. An execution sandbox may require permission to connect to this local service; ordinary terminal use does not.

## Analyze a photo

```sh
.venv/bin/fruit-analyze evaluation/images/banana-yellow.jpg
# Equivalent:
.venv/bin/python -m fruit_backend /path/to/photo.jpg
```

Success prints JSON to stdout with two keys: `assessment` and application-generated `metadata`. Redirect stdout to save a result if desired. Errors print a JSON object to stderr and return exit code 1; file errors return 2. A valid `no_fruit`, `multiple_fruits`, or `unclear_image` assessment returns 0, since inference succeeded.

Python API:

```python
from pathlib import Path
from fruit_backend import analyze_image, AnalysisError

try:
    result = analyze_image(Path("evaluation/images/banana-yellow.jpg").read_bytes())
    print(result.assessment.fruit_name)
    print(result.assessment.ripeness)
    print(result.model_dump_json(indent=2))
except AnalysisError as error:
    print(error.code, str(error))
```

For repeated requests, reuse `FruitAnalyzer()` and call its `analyze_image(bytes)` method. It checks model availability once per instance; every image gets a fresh chat. Neither the convenience function nor the reusable class caches predictions or persists user images.

## Output contract

| Assessment field | Values |
| --- | --- |
| `status` | `assessed`, `no_fruit`, `multiple_fruits`, `unclear_image` |
| `fruit_name` | Singular common English name, or null for non-assessed scenes |
| `ripeness` | `unripe`, `ripe`, `overripe`, `uncertain` |
| `visible_spoilage` | `none_observed`, `possible`, `clear`, `uncertain` |
| `observations` | One to four short visible observations |
| `explanation` | Brief assessment rationale or uncertainty explanation |

Non-assessed scenes require null fruit identification and uncertain condition fields. An identified fruit can still have uncertain ripeness. Overripeness and spoilage are independent. The backend does not produce an edible/not-edible verdict: visual appearance cannot establish food safety or internal condition.

The generation schema has separate assessed/rejected branches, so a rejection cannot carry a fruit name or definite condition fields. Pydantic validates the final answer again after inference. The compact prompt describes the task and fields; the complete schema is supplied through Ollama's `format` parameter rather than duplicated in the prompt.

Metadata includes the model tag and digest, prompt version/hash, original and prepared image hashes, prepared dimensions, elapsed/inference/model-load timing, token count, generation settings, and whether the model emitted internal thinking. Internal reasoning text is not returned or persisted by the backend.

Input supports JPEG, PNG, and WebP up to **10 MiB / 25 megapixels**. Animated, corrupted, and unsupported inputs are rejected. Preparation corrects EXIF orientation, composites transparency onto white, removes metadata, and downsizes to a maximum 1,280-pixel edge before JPEG encoding. One fruit is expected; one cut half or two halves of the same fruit can be assessed. Multiple separate fruits are rejected, including multiple fruits of one type.

## Inference configuration and failure handling

Defaults: temperature **0.6**, top-p 0.95, top-k 20, repetition penalty 1.0, presence penalty 0.0, seed 1234, context 8,192 tokens, output budget **4,096 tokens**, `stream=False`, `think=True`, and a five-minute keep-alive. Sampling parameters follow [Qwen's current thinking-model evaluation settings](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction), with a smaller local token budget. Thinking is explicitly enabled as requested. Ollama separates `message.thinking` from `message.content`; only `message.content` is parsed and validated as the final assessment. Thinking text is not included in the returned result or saved reports.

The initial 768-token budget truncated JSON in development because this installed thinking variant emitted reasoning even with `think=False`. A 2,048-token budget also truncated ambiguous/rejection cases with thinking explicitly enabled. The larger budget leaves space for both thinking and the final answer. Greedy decoding was replaced with Qwen's documented sampling settings. Separating the fields does not eliminate the token or latency cost of reasoning. The fixed seed improves reproducibility but does not guarantee identical or correct outputs across machines or runtime changes.

Connection timeout is 3 seconds; HTTP read timeout is 180 seconds to accommodate thinking. The latter is a network inactivity timeout, not a guaranteed hard wall-clock cancellation of Ollama computation. Requests are not automatically retried.

| Error code | Next action |
| --- | --- |
| `invalid_image` | Supply an intact, supported still image within the limits |
| `service_unavailable` | Start Ollama or check local connectivity |
| `model_missing` | Pull `qwen3-vl:4b` |
| `timeout` | Check system load/memory, then explicitly retry |
| `invalid_model_output` | Output was empty, truncated, or failed schema/consistency validation; explicitly retry |
| `inference_failed` | Inspect Ollama's error and available resources |

## Tests and evaluation

Deterministic tests do not require Ollama:

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m mypy
```

The real-model evaluator defaults to the development split and writes raw predictions, expected labels, per-image errors, timing, and summary counts. Existing reports are preserved unless `--overwrite` is explicitly passed. Partial results are saved after each image.

```sh
.venv/bin/fruit-evaluate --split development --output output/evaluation/my-development-run.json
.venv/bin/fruit-evaluate --split held_out --output output/evaluation/my-held-out-run.json
```

Do prompt development using the development split. Run held-out images after settings are fixed; do not repeatedly tune against those images while describing the results as untouched hold-out performance.

The 20 fixtures and their labels are in `evaluation/manifest.json`; credits, original links, license links, and crop/degradation details are in [evaluation/ATTRIBUTION.md](evaluation/ATTRIBUTION.md). Two fixtures are synthetic controls. Labels were assigned by the implementation assistant before evaluation and are **provisional visual judgments**, not independent expert or laboratory measurements. Uncertain reference cases are scored separately. Operational errors and abstentions on clear references remain in accuracy denominators.

Open [the evaluation notebook](output/jupyter-notebook/fruit-evaluation.ipynb) in a notebook editor with the project `.venv` selected, or execute it headlessly:

```sh
.venv/bin/python scripts/execute_notebook.py
```

By default the notebook inspects saved real-model reports without rerunning inference. To perform a fresh development evaluation from the notebook:

```sh
FRUIT_EVAL_LIVE=1 .venv/bin/python scripts/execute_notebook.py
```

See [the evaluation findings](evaluation/RESULTS.md) for observed performance and remaining limitations. Public photos may overlap model training data; new phone photos are the next meaningful check before building the interface.
