"""Run the documented synthetic smoke experiment without network access."""

from stellar_benchmark.config import SyntheticDemoConfig
from stellar_benchmark.demo import run_synthetic_demo


if __name__ == "__main__":
    artifacts = run_synthetic_demo(SyntheticDemoConfig())
    summary = artifacts["summary"].read_text(encoding="utf-8")
    print(summary)
    print(f"Saved artifacts to {artifacts['summary'].parent.resolve()}")
