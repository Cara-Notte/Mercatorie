from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.dataset_diagnostics.diagnostics import DatasetDiagnosticsResult


def write_diagnostics_outputs(result: DatasetDiagnosticsResult, output_dir: str | Path) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "dataset_diagnostics_summary.json"
    md_path = out_dir / "dataset_diagnostics_report.md"
    gap_csv_path = out_dir / "dataset_gap_metrics.csv"
    coverage_csv_path = out_dir / "dataset_feature_target_coverage.csv"
    windows_csv_path = out_dir / "dataset_training_windows.csv"
    class_balance_csv_path = out_dir / "dataset_class_balance.csv"
    match_quality_csv_path = out_dir / "dataset_match_quality.csv"

    json_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_build_markdown_report(result), encoding="utf-8")

    pd.DataFrame(result.gaps.get("by_commodity", [])).to_csv(gap_csv_path, index=False)

    features = pd.DataFrame(result.feature_coverage.get("feature_non_null_rates_by_commodity", []))
    targets = pd.DataFrame(result.target_coverage.get("by_commodity", []))
    
    if features.empty:
        merged = targets
    elif targets.empty:
        merged = features
    else:
        merged = features.merge(targets, on=["commodity", "row_count"], how="outer")

    merged.sort_values("commodity").to_csv(coverage_csv_path, index=False)

    pd.DataFrame(result.target_coverage.get("training_windows", [])).sort_values(
        ["commodity", "horizon"]
    ).to_csv(windows_csv_path, index=False)
    pd.DataFrame(result.class_balance.get("by_commodity", [])).sort_values(
        ["commodity", "horizon"]
    ).to_csv(class_balance_csv_path, index=False)
    pd.DataFrame(result.lag_target_match_quality.get("by_commodity", [])).sort_values(
        ["commodity", "field"]
    ).to_csv(match_quality_csv_path, index=False)

    return {
        "json": json_path,
        "markdown": md_path,
        "gap_csv": gap_csv_path,
        "coverage_csv": coverage_csv_path,
        "windows_csv": windows_csv_path,
        "class_balance_csv": class_balance_csv_path,
        "match_quality_csv": match_quality_csv_path,
    }


def _build_markdown_report(result: DatasetDiagnosticsResult) -> str:
    payload = result.to_dict()
    lines = [
        "# Dataset Diagnostics Report",
        "",
        f"**Readiness:** `{payload['readiness_status']}`",
        "",
        "## Failures",
    ]
    failures = payload.get("failures", [])
    if failures:
        lines.extend([f"- {item}" for item in failures])
    else:
        lines.append("- None")

    lines.extend(["", "## Warnings"])
    warnings = payload.get("warnings", [])
    if warnings:
        lines.extend([f"- {item}" for item in warnings])
    else:
        lines.append("- None")

    lines.extend(["", "## Canonical Summary"])
    canonical = payload.get("canonical", {})
    lines.append(f"- Overall rows: {canonical.get('row_count_overall', 0)}")
    lines.append(
        f"- Date range: {canonical.get('min_date_overall')} to {canonical.get('max_date_overall')}"
    )
    lines.append(
        f"- Duplicate (date, commodity) pairs: {canonical.get('duplicate_date_commodity_count', 0)}"
    )

    lines.extend(["", "## Horizon Trainability"])
    target_cov = payload.get("target_coverage", {})
    lines.append(f"- Trainable rows 7d (model usable): {target_cov.get('trainable_rows_7d_total', 0)}")
    lines.append(f"- Trainable rows 30d (model usable): {target_cov.get('trainable_rows_30d_total', 0)}")

    lines.extend(["", "## Feature Tier Summary"])
    tiers = payload.get("feature_coverage", {}).get("feature_tiers", {})
    for tier_name, tier_cols in tiers.items():
        lines.append(f"- {tier_name}: {', '.join(tier_cols)}")

    lines.extend(["", "## Class Balance (Overall)"])
    for horizon, rec in payload.get("class_balance", {}).get("overall", {}).items():
        lines.append(f"- {horizon}: counts={rec.get('class_counts', {})}, ratios={rec.get('class_ratios', {})}")

    lines.extend(["", "## Match Quality (Overall)"])
    for rec in payload.get("lag_target_match_quality", {}).get("overall", []):
        lines.append(
            "- {field}: exact={exact_ratio:.3f}, tolerance={tolerance_ratio:.3f}, no_match={no_match_ratio:.3f}".format(
                field=rec.get("field"),
                exact_ratio=rec.get("exact_ratio", 0.0),
                tolerance_ratio=rec.get("tolerance_ratio", 0.0),
                no_match_ratio=rec.get("no_match_ratio", 0.0),
            )
        )

    return "\n".join(lines) + "\n"
