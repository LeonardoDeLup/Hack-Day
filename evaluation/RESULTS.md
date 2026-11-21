# Evaluation findings

The backend uses the existing local `qwen3-vl:4b` model, digest `a1b29c246319cdcb1f3842f36d29f198f737c0cf516ba1bcba34b1b974e87980`, on this Apple Silicon Mac with Ollama 0.40.17. No model weights were trained or replaced.

## Reference set and interpretation

The frozen manifest contains 20 cases: 13 development and 7 held-out. There are three examples each of bananas, apples, oranges, strawberries, and avocados, plus five scope/quality/spoilage challenges. Two challenges are synthetic controls: a geometric checkerboard and a severely degraded banana image. See [image credits and transformations](ATTRIBUTION.md).

Labels were assigned by the implementation assistant from visible appearance before evaluation. They are provisional, not independent expert or laboratory measurements. There are 16 identifiable-fruit references, including 11 with an assessable visual ripeness label and five labeled uncertain. The remaining four are scene-rejection cases. The public photos may overlap training data, and the small, curated set is not representative of arbitrary phone uploads.

The reference manifest SHA-256 is `7b654353010e6f373ae8e53a66680828174747740c553e1cbdd1411202a3fe31`. Image hashes are checked before every evaluation. Labels and splits were not changed to fit predictions.

## Development findings

- The initial smoke test identified a yellow banana as ripe and produced valid JSON in 17.9 seconds, including 2.8 seconds of model loading. Its earlier prompt/settings are recorded in `output/evaluation/baseline-banana-v1.json`.
- At a 768-token budget, the first full development run had 10 invalid responses out of 13 cases. The installed thinking variant generated internal reasoning even when `think=False` was requested, leaving too few tokens for the final JSON.
- A 2,048-token run eliminated truncation on several clear fruit examples, but mislabeled a green apple as unripe, a cut strawberry as overripe, and a green orange as a lime. That run was interrupted after eight cases when thinking was explicitly enabled at the user's request; its partial report is retained.
- The revised prompt addressed the apple-color and cut-surface interpretation errors. With thinking enabled, greedy decoding, and 2,048 tokens, all nine completed clear-fruit cases matched their reference fruit/ripeness labels; one citrus and three rejection cases failed operationally. Those failures remain in the report's denominators.
- A separate greedy-decoding citrus probe also exhausted 4,096 tokens. Increasing the budget alone was not sufficient for that attempt.
- Final sampling parameters use temperature 0.6, top-p 0.95, top-k 20, repetition penalty 1.0, presence penalty 0.0, and seed 1234, following [Qwen's current thinking-model evaluation settings](https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction). The local budget is capped at 4,096 generated tokens and an 8,192-token context. A sampled checkerboard probe correctly returned `no_fruit` in 63.0 seconds using 2,357 tokens.
- Sampling alone did not solve every problem: the longer-prompt development run still exhausted 4,096 tokens for a cut strawberry and a bunch of bananas, and misidentified the green orange as a guava. Its 12-case partial report is retained as `development-v3-sampling-partial.json`.
- A compact prompt, without the full schema duplicated in the user message, produced complete answers on difficult probes. Runtime validation then exposed inconsistent rejection fields. The final `fruit-v5` request uses separate assessed/rejected JSON-schema branches to enforce these relationships during generation, while preserving the Python validator. A multiple-banana probe then correctly returned the constrained rejection in 5.3 seconds / 206 generated tokens with thinking enabled.

Thinking is separate from the final answer in Ollama. Only `message.content` is parsed as the assessment; `message.thinking` is neither exposed by the backend nor stored in evaluation reports. Its presence is recorded as a boolean. Separating these fields does not remove thinking's token or latency cost.

## Final evaluation

The final `fruit-v5` run completed **17 of 20 cases** before stopping at the user’s request: all 13 development cases and 4 of 7 held-out cases. All 17 completed calls returned valid structured results with **zero operational errors**. No further tuning was performed on held-out predictions.

| Measure | Development | Completed held-out subset |
| --- | --- | --- |
| Fruit identification | 9/10 | 3/4 |
| Correct fruit and assessable ripeness | 8/10 | 1/1 |
| Appropriate uncertainty on ambiguous ripeness | No reference cases | 0/3 |
| Correct fruit and visible spoilage label | 7/10 | 1/4 |
| Exact scene-rejection label | 2/3 | No reference cases |

The three unrun held-out cases are `avocado-tree`, `decomposed-apple`, and `multiple-citrus-mold`. The partial held-out sample is too small for a reliable general accuracy estimate; in particular, 1/1 ripeness agreement is not evidence of perfect ripeness performance.

Notable remaining errors: the dark banana was labeled ripe rather than overripe; the green orange was identified as mango; the moldy orange was identified as lime; and the unreadable image was rejected as no-fruit rather than unclear-image. The model also gave definite ripeness labels for all three completed reference-uncertain held-out examples. These are model-quality limitations, not parsing failures.

Development latency: 8.5 seconds for the first call, 10.3 seconds median for subsequent calls, and 104.3 seconds maximum. These are observed request times, not guaranteed cold/warm timings. The held-out subset ranged from approximately 12.6 to 47.7 seconds.

Raw results are in `output/evaluation/development-final.json` and `held-out-final.json`. `final-evaluation.json` combines them and explicitly marks the early stop. The evaluation notebook replays these real-model results without additional inference by default. Backend verification: 57 deterministic tests passed; the separate opt-in live test was skipped because the real-model evaluations already exercised the backend. Strict type checking passed.

The backend is usable for the hackathon, but these results do not establish dependable ripeness or spoilage judgments across arbitrary images.

## Limits to preserve in future work

- Green citrus can be visually ambiguous between an orange and a lime. A confident fruit name can still be wrong.
- Color alone is not a general ripeness test, especially for green apple varieties and intact avocados.
- Spoilage and ripeness are separate; decay or damage does not establish maturity.
- Operational success and valid JSON do not establish predictive correctness.
- Evaluate new phone photos under actual demo lighting before making broad accuracy claims. Keep any future prompt tuning separate from a fresh held-out set.
