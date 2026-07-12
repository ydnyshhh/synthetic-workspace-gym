# Deterministic counterfactual example

Generate this example with the quick-start commands in `docs/counterfactual-branching.md`. The checked-in demo pack includes trusted-only hidden evaluator assets so it can be replayed locally. The top-level SFT, preference, critic, and RL records are schema-only illustrative examples; the files under `demo-run/` are produced by executing the checked-in pack. Hidden assets must never be exposed through model-facing tools or messages.


## Positive-regret evaluator demo

Run `uv run python examples/counterfactual/positive_demo.py`. The generated `positive-demo/summary.json` records an original return of `0.0`, corrected return of `1.0`, regret of `1.0`, and `recoverable: true`. The corrected action is intentionally privileged and exists only to validate the complete causal-analysis pipeline.
