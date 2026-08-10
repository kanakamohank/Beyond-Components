"""
Experiment history logger — append-only JSONL file for tracking all runs.

Every experiment run appends a single JSON line to a persistent history file.
This ensures no results are ever lost, even if individual result files are
moved or deleted.

Usage::

    from src.analysis.experiment_history import append_experiment_result

    append_experiment_result(
        experiment_type="fourier_discovery",
        model_key="pythia-1.4b",
        results_summary={...},
        config={...},
        output_path="fourier_results/...",
    )

The history file location defaults to ``experiment_history.jsonl`` in the
project root, configurable via the ``EXPERIMENT_HISTORY_FILE`` env var.
"""

import datetime
import json
import logging
import os
import platform
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default history file in project root
_DEFAULT_HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiment_history.jsonl",
)

HISTORY_FILE = os.environ.get("EXPERIMENT_HISTORY_FILE", _DEFAULT_HISTORY_FILE)


def _numpy_to_python(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(i) for i in obj]
    elif isinstance(obj, float) and (np.isinf(obj) or np.isnan(obj)):
        return str(obj)
    return obj


def append_experiment_result(
    experiment_type: str,
    model_key: str,
    results_summary: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    output_path: Optional[str] = None,
    notes: Optional[str] = None,
    history_file: Optional[str] = None,
) -> str:
    """Append a single experiment result to the persistent history file.

    Args:
        experiment_type: Type of experiment (e.g. ``"fourier_discovery"``,
            ``"mask_learning"``, ``"causal_validation"``).
        model_key: Model registry key.
        results_summary: Key metrics and findings (will be JSON-serialized).
        config: Full experiment configuration (optional).
        output_path: Path to the detailed results file (optional).
        notes: Free-text notes about the run (optional).
        history_file: Override the default history file path.

    Returns:
        Path to the history file.
    """
    filepath = history_file or HISTORY_FILE

    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "experiment_type": experiment_type,
        "model_key": model_key,
        "results_summary": _numpy_to_python(results_summary),
        "output_path": output_path,
        "notes": notes,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
    }
    if config is not None:
        record["config"] = _numpy_to_python(config)

    # Atomic-ish append: open in append mode, write one line, flush
    try:
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"Experiment result appended to {filepath}")
    except Exception as e:
        logger.error(f"Failed to write experiment history: {e}")
        # Don't crash the experiment over a logging failure
        raise

    return filepath


def load_experiment_history(
    history_file: Optional[str] = None,
    experiment_type: Optional[str] = None,
    model_key: Optional[str] = None,
) -> list:
    """Load experiment history, optionally filtered by type or model.

    Args:
        history_file: Override the default history file path.
        experiment_type: Filter by experiment type.
        model_key: Filter by model key.

    Returns:
        List of experiment records (dicts).
    """
    filepath = history_file or HISTORY_FILE

    if not os.path.exists(filepath):
        return []

    records = []
    with open(filepath, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(f"Skipping malformed line {line_num} in {filepath}")
                continue

            if experiment_type and record.get("experiment_type") != experiment_type:
                continue
            if model_key and record.get("model_key") != model_key:
                continue
            records.append(record)

    return records


def print_experiment_history(
    history_file: Optional[str] = None,
    experiment_type: Optional[str] = None,
    model_key: Optional[str] = None,
    last_n: Optional[int] = None,
) -> str:
    """Print a formatted table of experiment history.

    Args:
        history_file: Override the default history file path.
        experiment_type: Filter by experiment type.
        model_key: Filter by model key.
        last_n: Only show the last N records.

    Returns:
        Formatted string.
    """
    records = load_experiment_history(history_file, experiment_type, model_key)

    if last_n:
        records = records[-last_n:]

    if not records:
        return "No experiment records found."

    lines = []
    lines.append("=" * 80)
    lines.append("EXPERIMENT HISTORY")
    lines.append("=" * 80)
    lines.append(f"{'#':>3} {'Timestamp':<20} {'Type':<20} {'Model':<15} {'Key Result'}")
    lines.append("-" * 80)

    for i, rec in enumerate(records, 1):
        ts = rec.get("timestamp", "?")[:19]
        exp_type = rec.get("experiment_type", "?")
        model = rec.get("model_key", "?")
        summary = rec.get("results_summary", {})

        # Extract a short key result string
        if "best_layer" in summary:
            key_result = (
                f"L{summary['best_layer']}: freq={summary.get('best_freq', '?')}, "
                f"ratio={summary.get('best_ratio', '?'):.1f}x"
            )
        elif "n_layers_analyzed" in summary:
            key_result = f"{summary['n_layers_analyzed']} layers analyzed"
        else:
            key_result = str(summary)[:40]

        lines.append(f"{i:>3} {ts:<20} {exp_type:<20} {model:<15} {key_result}")

    lines.append("=" * 80)
    return "\n".join(lines)
