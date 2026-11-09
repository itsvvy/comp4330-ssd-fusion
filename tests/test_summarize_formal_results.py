import json

from scripts.summarize_formal_results import markdown, summarize


ARMS = ("m0_ft", "control", "m1_fusion")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_summarize_latest_formal_runs(tmp_path):
    run_root = tmp_path / "runs"
    eval_root = tmp_path / "eval"
    for index, arm in enumerate(ARMS):
        run_dir = run_root / arm / f"2026010{index + 1}T000000Z"
        eval_dir = eval_root / arm / run_dir.name
        write_json(
            run_dir / "model_info.json",
            {
                "variant": arm,
                "total_parameters": 100 + index,
                "trainable_parameters": 50 + index,
                "control_channels": None,
            },
        )
        write_json(
            eval_dir / "metrics.json",
            {
                "map_0.5": 0.70 + index / 100,
                "fps": 20.0,
                "latency_ms_mean": 50.0,
            },
        )
        write_json(
            eval_dir / "size_metrics.json",
            {
                "overall": {"ap": 0.71, "recall": 0.9},
                "size_bins": {
                    "small": {"ap": 0.1, "recall": 0.7},
                    "medium": {"ap": 0.6, "recall": 0.9},
                    "large": {"ap": 0.8, "recall": 0.95},
                },
            },
        )

    summary = summarize(run_root, eval_root)
    assert set(summary) == set(ARMS)
    assert summary["m1_fusion"]["overall_map_0.5"] == 0.72
    assert summary["control"]["size_bins"]["small"]["ap"] == 0.1


def test_markdown_includes_recall_and_ablation_deltas(tmp_path):
    run_root = tmp_path / "runs"
    eval_root = tmp_path / "eval"
    for index, arm in enumerate(ARMS):
        run_dir = run_root / arm / "20260101T000000Z"
        eval_dir = eval_root / arm / run_dir.name
        write_json(
            run_dir / "model_info.json",
            {
                "variant": arm,
                "total_parameters": 100 + index,
                "trainable_parameters": 50 + index,
                "control_channels": None,
            },
        )
        write_json(
            eval_dir / "metrics.json",
            {
                "map_0.5": 0.70 + index / 100,
                "fps": 20.0,
                "latency_ms_mean": 50.0,
            },
        )
        write_json(
            eval_dir / "size_metrics.json",
            {
                "overall": {"ap": 0.71, "recall": 0.9},
                "size_bins": {
                    "small": {"ap": 0.10 + index / 100, "recall": 0.70},
                    "medium": {"ap": 0.60 + index / 100, "recall": 0.90},
                    "large": {"ap": 0.80 + index / 100, "recall": 0.95},
                },
            },
        )

    rendered = markdown(summarize(run_root, eval_root))
    assert "| Arm | Small recall |" in rendered
    assert "| M1 - C | +0.0100 | +0.0100 | +0.0100 | +0.0100 |" in rendered
    assert "| C - M0-FT | +0.0100 | +0.0100 | +0.0100 | +0.0100 |" in rendered
