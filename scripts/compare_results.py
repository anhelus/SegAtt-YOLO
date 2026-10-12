#!/usr/bin/env python3
"""
Compare training results across models.

Scans runs/<dataset>/<model>/results.csv, extracts best epoch metrics,
and outputs a comparison table (CSV + Excel).

Usage:
    python scripts/compare_results.py --config viz_config.yaml --dataset tomatoes
    python scripts/compare_results.py --runs-dir runs/complete --dataset tomatoes --output reports/comparison/tomatoes_comparison
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import yaml

# Metriche standard da confrontare
DEFAULT_METRICS = [
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
]


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def find_results_csvs(runs_dir: Path, dataset: str) -> list[tuple[str, Path]]:
    """
    Find all results.csv files for a dataset.
    Returns list of (model_name, csv_path).
    """
    dataset_dir = runs_dir / dataset
    if not dataset_dir.exists():
        return []

    results = []
    for model_dir in dataset_dir.iterdir():
        if not model_dir.is_dir():
            continue
        csv_path = model_dir / "results.csv"
        if csv_path.exists():
            results.append((model_dir.name, csv_path))

    return results


def extract_best_metrics(csv_path: Path, metrics: list[str]) -> dict:
    """
    Read results.csv and return best epoch metrics (last row).
    """
    try:
        df = pl.read_csv(csv_path)
        if df.height == 0:
            return {}

        # Last row = best epoch (YOLO saves best at end)
        last_row = df.row(df.height - 1, named=True)

        # Extract requested metrics
        extracted = {}
        for m in metrics:
            if m in last_row:
                extracted[m] = last_row[m]

        return extracted
    except Exception as e:
        print(f"  [WARN] Error reading {csv_path}: {e}", file=sys.stderr)
        return {}


def build_comparison_table(
    runs_dir: Path,
    dataset: str,
    metrics: list[str],
    labels: Optional[dict] = None,
    baseline: Optional[str] = None,
) -> pl.DataFrame:
    """
    Build comparison DataFrame across all models for a dataset.
    """
    csvs = find_results_csvs(runs_dir, dataset)
    if not csvs:
        print(f"[ERROR] No results.csv found in {runs_dir / dataset}")
        return pl.DataFrame()

    rows = []
    for model_name, csv_path in csvs:
        metrics_dict = extract_best_metrics(csv_path, metrics)
        if not metrics_dict:
            print(f"  [WARN] {model_name}: no metrics extracted")
            continue

        row = {"model": model_name}
        if labels and model_name in labels:
            row["label"] = labels[model_name]
        if baseline:
            row["is_baseline"] = (model_name == baseline)
        row.update(metrics_dict)
        rows.append(row)

    if not rows:
        return pl.DataFrame()

    df = pl.DataFrame(rows)

    # Add delta vs baseline if specified
    if baseline and baseline in df["model"].to_list():
        baseline_row = df.filter(pl.col("model") == baseline).row(0, named=True)
        for m in metrics:
            if m in df.columns and m in baseline_row:
                base_val = baseline_row[m]
                if base_val is not None and base_val != 0:
                    df = df.with_columns(
                        ((pl.col(m) - base_val) / abs(base_val) * 100).alias(f"{m}_delta_pct")
                    )

    return df


def sort_by_metric(df: pl.DataFrame, metric: str = "metrics/mAP50-95(B)", descending: bool = True) -> pl.DataFrame:
    """Sort dataframe by a metric."""
    if metric in df.columns:
        return df.sort(metric, descending=descending)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare YOLO training results across models."
    )
    parser.add_argument(
        "--config", type=str, default=None, help="viz_config.yaml path (for labels/baseline/metrics)"
    )
    parser.add_argument("--runs-dir", type=str, default="runs/complete", help="Root runs directory")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., tomatoes)")
    parser.add_argument(
        "--output", type=str, default=None, help="Output prefix (without extension)"
    )
    parser.add_argument("--metrics", type=str, nargs="+", default=None, help="Metrics to compare")
    parser.add_argument(
        "--sort-by", type=str, default="metrics/mAP50-95(B)", help="Metric to sort by"
    )
    parser.add_argument("--no-excel", action="store_true", help="Skip Excel output")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)

    # Load config if provided
    config = {}
    if args.config:
        config = load_config(args.config)
        ds_cfg = config.get("datasets", {}).get(args.dataset, {})
        labels = ds_cfg.get("labels", {})
        baseline = ds_cfg.get("baseline")
        metrics = args.metrics or ds_cfg.get("metrics", DEFAULT_METRICS)
    else:
        labels = {}
        baseline = None
        metrics = args.metrics or DEFAULT_METRICS

    print(f"Comparing results for dataset: {args.dataset}")
    print(f"Runs dir: {runs_dir}")
    print(f"Metrics: {metrics}")

    # Build comparison table
    df = build_comparison_table(runs_dir, args.dataset, metrics, labels, baseline)

    if df.height == 0:
        print("[ERROR] No results found.")
        return 1

    # Sort
    df = sort_by_metric(df, args.sort_by)

    # Pretty print
    print("\n" + "=" * 100)
    print(f"RESULTS COMPARISON - {args.dataset.upper()}")
    print("=" * 100)

    # Select columns to display (model, label, key metrics, deltas)
    display_cols = ["model"]
    if "label" in df.columns:
        display_cols.append("label")
    if "is_baseline" in df.columns:
        display_cols.append("is_baseline")

    # Add main metrics
    for m in ["metrics/mAP50(B)", "metrics/mAP50-95(B)", "metrics/precision(B)", "metrics/recall(B)"]:
        if m in df.columns:
            display_cols.append(m)

    # Add delta columns
    for c in df.columns:
        if c.endswith("_delta_pct"):
            display_cols.append(c)

    # Print table
    try:
        # Use polars string representation
        print(df.select(display_cols))
    except Exception:
        # Fallback: print raw rows
        for row in df.select(display_cols).iter_rows(named=True):
            print(row)

    # Output paths
    output_prefix = args.output or f"reports/comparison/{args.dataset}_comparison"

    # CSV
    csv_path = f"{output_prefix}.csv"
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(csv_path)
    print(f"[OK] CSV saved: {csv_path}")

    # Excel
    if not args.no_excel:
        try:
            excel_path = f"{output_prefix}.xlsx"
            df.write_excel(excel_path)
            print(f"[OK] Excel saved: {excel_path}")
        except ImportError:
            print("[WARN] Excel output skipped (install openpyxl: pip install openpyxl)")

    return 0


if __name__ == "__main__":
    sys.exit(main())