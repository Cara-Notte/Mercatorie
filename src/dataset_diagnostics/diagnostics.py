from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd

from src.dataset_diagnostics.config import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    DiagnosticsConfig,
)


class ReadinessStatus(str, Enum):
    PASS = "PASS"
    PASS_WITH_WARNINGS = "PASS_WITH_WARNINGS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class DatasetDiagnosticsResult:
    readiness_status: ReadinessStatus
    failures: list[str]
    warnings: list[str]
    canonical: dict[str, Any]
    gaps: dict[str, Any]
    feature_coverage: dict[str, Any]
    target_coverage: dict[str, Any]
    distributions: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["readiness_status"] = self.readiness_status.value
        return payload


def _serialize_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(_serialize_scalar)
    return out.to_dict(orient="records")


def run_dataset_diagnostics(
    canonical_df: pd.DataFrame,
    training_ready_df: pd.DataFrame,
    config: DiagnosticsConfig | None = None,
) -> DatasetDiagnosticsResult:
    cfg = config or DiagnosticsConfig()
    canonical = _canonical_diagnostics(canonical_df, cfg)
    gaps = _gap_diagnostics(canonical_df)
    feature_coverage = _feature_coverage_diagnostics(training_ready_df)
    target_coverage = _target_diagnostics(training_ready_df)
    distributions = _distribution_diagnostics(training_ready_df, cfg)

    readiness_status, failures, warnings = evaluate_readiness(
        canonical=canonical,
        gaps=gaps,
        feature_coverage=feature_coverage,
        target_coverage=target_coverage,
        distributions=distributions,
        config=cfg,
    )

    return DatasetDiagnosticsResult(
        readiness_status=readiness_status,
        failures=failures,
        warnings=warnings,
        canonical=canonical,
        gaps=gaps,
        feature_coverage=feature_coverage,
        target_coverage=target_coverage,
        distributions=distributions,
    )


def _canonical_diagnostics(df: pd.DataFrame, config: DiagnosticsConfig) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["commodity", "date"]).reset_index(drop=True)

    grouped = (
        frame.groupby("commodity", as_index=False)
        .agg(rows=("commodity", "size"), min_date=("date", "min"), max_date=("date", "max"))
        .sort_values("commodity")
    )

    duplicate_mask = frame.duplicated(subset=["date", "commodity"], keep=False)
    duplicate_pairs = frame.loc[duplicate_mask, ["date", "commodity"]].drop_duplicates()

    null_price_count = int(frame["price_idr"].isna().sum())
    invalid_price_count = int((~pd.to_numeric(frame["price_idr"], errors="coerce").notna()).sum())
    non_positive_price_count = int((pd.to_numeric(frame["price_idr"], errors="coerce") <= 0).fillna(False).sum())

    unexpected_commodities: list[str] = []
    if config.allowed_commodities is not None:
        observed = set(frame["commodity"].dropna().astype(str))
        expected = set(config.allowed_commodities)
        unexpected_commodities = sorted(observed - expected)

    return {
        "row_count_overall": int(len(frame)),
        "row_count_by_commodity": _records(grouped[["commodity", "rows"]]),
        "min_date_overall": _serialize_scalar(frame["date"].min()),
        "max_date_overall": _serialize_scalar(frame["date"].max()),
        "date_span_by_commodity": _records(grouped[["commodity", "min_date", "max_date"]]),
        "duplicate_date_commodity_count": int(len(duplicate_pairs)),
        "duplicate_date_commodity_examples": _records(duplicate_pairs.head(20)),
        "null_price_idr_count": null_price_count,
        "invalid_price_idr_count": invalid_price_count,
        "non_positive_price_count": non_positive_price_count,
        "unexpected_commodities": unexpected_commodities,
    }


def _gap_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    rows: list[dict[str, Any]] = []
    for commodity, part in frame.groupby("commodity", sort=True):
        dates = part["date"].dropna().sort_values().drop_duplicates()
        if dates.empty:
            continue
        day_span = max((dates.iloc[-1] - dates.iloc[0]).days + 1, 1)
        obs_count = int(len(dates))
        missing_ratio = float((day_span - obs_count) / day_span)

        gaps = dates.diff().dropna().dt.days
        median_gap = float(gaps.median()) if not gaps.empty else 0.0
        p90_gap = float(gaps.quantile(0.9)) if not gaps.empty else 0.0
        max_gap = int(gaps.max()) if not gaps.empty else 0

        rows.append(
            {
                "commodity": commodity,
                "observation_count": obs_count,
                "span_days": int(day_span),
                "missing_day_ratio": missing_ratio,
                "gap_median_days": median_gap,
                "gap_p90_days": p90_gap,
                "gap_max_days": max_gap,
                "gap_gt_1d_count": int((gaps > 1).sum()),
                "gap_gt_3d_count": int((gaps > 3).sum()),
                "gap_gt_7d_count": int((gaps > 7).sum()),
                "gap_gt_30d_count": int((gaps > 30).sum()),
            }
        )

    table = pd.DataFrame(rows).sort_values("commodity") if rows else pd.DataFrame()
    return {"by_commodity": _records(table)}


def _feature_coverage_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    coverage_rows: list[dict[str, Any]] = []
    first_valid_rows: list[dict[str, Any]] = []

    for commodity, part in frame.groupby("commodity", sort=True):
        total = max(len(part), 1)
        row: dict[str, Any] = {"commodity": commodity, "row_count": int(len(part))}
        for feature_col in FEATURE_COLUMNS:
            if feature_col not in part.columns:
                row[f"{feature_col}_non_null_rate"] = 0.0
                first_valid = None
            else:
                non_null = part[feature_col].notna().sum()
                row[f"{feature_col}_non_null_rate"] = float(non_null / total)
                valid_dates = part.loc[part[feature_col].notna(), "date"]
                first_valid = valid_dates.min() if not valid_dates.empty else None
            first_valid_rows.append(
                {
                    "commodity": commodity,
                    "feature": feature_col,
                    "first_valid_date": _serialize_scalar(first_valid),
                }
            )

        usable_mask = part[list(FEATURE_COLUMNS)].notna().all(axis=1)
        row["fully_usable_feature_rows"] = int(usable_mask.sum())
        coverage_rows.append(row)

    return {
        "feature_non_null_rates_by_commodity": coverage_rows,
        "feature_first_valid_date_by_commodity": first_valid_rows,
    }


def _target_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()

    rows: list[dict[str, Any]] = []
    for commodity, part in frame.groupby("commodity", sort=True):
        total = max(len(part), 1)
        feat_valid = part[list(FEATURE_COLUMNS)].notna().all(axis=1)
        target_7d_valid = part["target_7d_inflation_pct"].notna()
        target_30d_valid = part["target_30d_inflation_pct"].notna()

        rows.append(
            {
                "commodity": commodity,
                "row_count": int(len(part)),
                "target_7d_non_null_rate": float(target_7d_valid.sum() / total),
                "target_30d_non_null_rate": float(target_30d_valid.sum() / total),
                "usable_target_7d_count": int(target_7d_valid.sum()),
                "usable_target_30d_count": int(target_30d_valid.sum()),
                "features_and_target_7d_overlap_count": int((feat_valid & target_7d_valid).sum()),
                "features_and_target_30d_overlap_count": int((feat_valid & target_30d_valid).sum()),
                "trainable_rows_7d": int((feat_valid & target_7d_valid).sum()),
                "trainable_rows_30d": int((feat_valid & target_30d_valid).sum()),
            }
        )

    table = pd.DataFrame(rows).sort_values("commodity") if rows else pd.DataFrame()
    return {
        "by_commodity": _records(table),
        "trainable_rows_7d_total": int(table["trainable_rows_7d"].sum()) if not table.empty else 0,
        "trainable_rows_30d_total": int(table["trainable_rows_30d"].sum()) if not table.empty else 0,
    }


def _summary_stats(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0, "min": None, "p50": None, "p90": None, "max": None, "mean": None}
    return {
        "count": int(clean.size),
        "min": float(clean.min()),
        "p50": float(clean.quantile(0.5)),
        "p90": float(clean.quantile(0.9)),
        "max": float(clean.max()),
        "mean": float(clean.mean()),
    }


def _distribution_diagnostics(df: pd.DataFrame, config: DiagnosticsConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for commodity, part in df.groupby("commodity", sort=True):
        rec = {
            "commodity": commodity,
            "price_idr": _summary_stats(part["price_idr"]),
            "price_change_1d_pct": _summary_stats(part.get("price_change_1d_pct", pd.Series(dtype=float))),
            "target_7d_inflation_pct": _summary_stats(
                part.get("target_7d_inflation_pct", pd.Series(dtype=float))
            ),
            "target_30d_inflation_pct": _summary_stats(
                part.get("target_30d_inflation_pct", pd.Series(dtype=float))
            ),
        }
        if config.enable_outlier_flags:
            rec["suspicious_price_jump_count"] = int(
                part.get("price_change_1d_pct", pd.Series(dtype=float)).abs().gt(100).sum()
            )
            rec["extreme_target_7d_count"] = int(
                part.get("target_7d_inflation_pct", pd.Series(dtype=float)).abs().gt(100).sum()
            )
            rec["extreme_target_30d_count"] = int(
                part.get("target_30d_inflation_pct", pd.Series(dtype=float)).abs().gt(100).sum()
            )
        rows.append(rec)

    return {"by_commodity": rows}


def evaluate_readiness(
    canonical: dict[str, Any],
    gaps: dict[str, Any],
    feature_coverage: dict[str, Any],
    target_coverage: dict[str, Any],
    distributions: dict[str, Any],
    config: DiagnosticsConfig,
) -> tuple[ReadinessStatus, list[str], list[str]]:
    t = config.thresholds
    failures: list[str] = []
    warnings: list[str] = []

    total_rows = max(canonical["row_count_overall"], 1)
    if canonical["duplicate_date_commodity_count"] > t.max_duplicate_pairs:
        failures.append("Duplicate (date, commodity) pairs exceed threshold.")

    null_ratio = canonical["null_price_idr_count"] / total_rows
    if null_ratio > t.max_null_price_ratio:
        failures.append(f"Null price ratio too high: {null_ratio:.3f}.")

    non_pos_ratio = canonical["non_positive_price_count"] / total_rows
    if non_pos_ratio > t.max_non_positive_price_ratio:
        failures.append(f"Non-positive price ratio too high: {non_pos_ratio:.3f}.")

    if canonical.get("unexpected_commodities"):
        failures.append("Unexpected commodities detected in canonical dataset.")

    for row in gaps.get("by_commodity", []):
        ratio = float(row["missing_day_ratio"])
        if ratio > t.max_missing_day_ratio_fail:
            failures.append(
                f"{row['commodity']}: missing-day ratio {ratio:.3f} exceeds fail threshold."
            )
        elif ratio > t.max_missing_day_ratio_warn:
            warnings.append(
                f"{row['commodity']}: missing-day ratio {ratio:.3f} exceeds warning threshold."
            )

    for row in feature_coverage.get("feature_non_null_rates_by_commodity", []):
        for feature_col in FEATURE_COLUMNS:
            rate = float(row.get(f"{feature_col}_non_null_rate", 0.0))
            if rate < t.min_feature_non_null_rate_warn:
                warnings.append(
                    f"{row['commodity']}: low non-null rate for {feature_col} ({rate:.3f})."
                )

    for row in target_coverage.get("by_commodity", []):
        for col in TARGET_COLUMNS:
            rate = float(row.get(f"{col.replace('_inflation_pct', '')}_non_null_rate", 0.0))
            if rate < t.min_target_non_null_rate_warn:
                warnings.append(f"{row['commodity']}: low non-null rate for {col} ({rate:.3f}).")

    if target_coverage.get("trainable_rows_7d_total", 0) < t.min_trainable_rows_7d:
        failures.append("Insufficient total trainable rows for 7d horizon.")
    if target_coverage.get("trainable_rows_30d_total", 0) < t.min_trainable_rows_30d:
        failures.append("Insufficient total trainable rows for 30d horizon.")

    for row in distributions.get("by_commodity", []):
        p90_abs_change = abs((row.get("price_change_1d_pct") or {}).get("p90") or 0.0)
        if p90_abs_change > t.max_abs_price_change_pct_fail:
            failures.append(f"{row['commodity']}: extreme p90 1d price change ({p90_abs_change:.2f}%).")
        elif p90_abs_change > t.max_abs_price_change_pct_warn:
            warnings.append(f"{row['commodity']}: elevated p90 1d price change ({p90_abs_change:.2f}%).")

    if failures:
        return ReadinessStatus.FAIL, sorted(set(failures)), sorted(set(warnings))
    if warnings:
        return ReadinessStatus.PASS_WITH_WARNINGS, [], sorted(set(warnings))
    return ReadinessStatus.PASS, [], []
