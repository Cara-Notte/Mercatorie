from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import pandas as pd

from src.dataset_builder.time_horizon_config import DEFAULT_LAG_CONFIGS, DEFAULT_TARGET_CONFIGS
from src.dataset_diagnostics.config import (
    CLASS_LABELS,
    FEATURE_COLUMNS,
    FEATURE_TIERS,
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
    class_balance: dict[str, Any]
    lag_target_match_quality: dict[str, Any]
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
    if df.empty:
        return []
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
    target_coverage = _target_diagnostics(training_ready_df, feature_coverage)
    class_balance = _class_balance_diagnostics(training_ready_df, feature_coverage, cfg)
    lag_target_match_quality = _lag_target_match_quality_diagnostics(training_ready_df)
    distributions = _distribution_diagnostics(training_ready_df, cfg)

    readiness_status, failures, warnings = evaluate_readiness(
        canonical=canonical,
        gaps=gaps,
        feature_coverage=feature_coverage,
        target_coverage=target_coverage,
        class_balance=class_balance,
        lag_target_match_quality=lag_target_match_quality,
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
        class_balance=class_balance,
        lag_target_match_quality=lag_target_match_quality,
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

        strict_mask = (
            part[list(FEATURE_COLUMNS)].notna().all(axis=1)
            if set(FEATURE_COLUMNS).issubset(part.columns)
            else pd.Series([False] * len(part), index=part.index)
        )
        critical_features = [c for c in FEATURE_TIERS["CRITICAL"] if c in part.columns]
        model_mask = (
            part[critical_features].notna().all(axis=1)
            if critical_features
            else pd.Series([False] * len(part), index=part.index)
        )

        row["strict_feature_usable_rows"] = int(strict_mask.sum())
        row["strict_feature_usable_rate"] = float(strict_mask.mean()) if len(part) else 0.0
        row["model_feature_usable_rows"] = int(model_mask.sum())
        row["model_feature_usable_rate"] = float(model_mask.mean()) if len(part) else 0.0

        for tier_name, tier_cols in FEATURE_TIERS.items():
            tier_present = [c for c in tier_cols if c in part.columns]
            if not tier_present:
                row[f"{tier_name.lower()}_all_non_null_rate"] = 0.0
                continue
            tier_mask = part[tier_present].notna().all(axis=1)
            row[f"{tier_name.lower()}_all_non_null_rate"] = float(tier_mask.mean())
            for col in tier_present:
                row[f"{tier_name.lower()}_{col}_non_null_rate"] = float(part[col].notna().mean())

        coverage_rows.append(row)

        coverage_df = pd.DataFrame(coverage_rows).sort_values("commodity") if coverage_rows else pd.DataFrame()

    return {
        "feature_tiers": {k: list(v) for k, v in FEATURE_TIERS.items()},
        "feature_non_null_rates_by_commodity": _records(coverage_df),
        "feature_first_valid_date_by_commodity": first_valid_rows,
        "strict_feature_usable_rows_total": int(coverage_df.get("strict_feature_usable_rows", pd.Series(dtype=int)).sum()) if not coverage_df.empty else 0,
        "model_feature_usable_rows_total": int(coverage_df.get("model_feature_usable_rows", pd.Series(dtype=int)).sum()) if not coverage_df.empty else 0,
    }


def _training_window(part: pd.DataFrame, model_mask: pd.Series, target_col: str) -> dict[str, Any]:
    usable = model_mask & part[target_col].notna()
    usable_dates = part.loc[usable, "date"]
    if usable_dates.empty:
        return {
            "first_usable_training_date": None,
            "last_usable_training_date": None,
            "usable_training_row_count": 0,
            "usable_training_span_days": 0,
        }

    first_date = usable_dates.min()
    last_date = usable_dates.max()
    return {
        "first_usable_training_date": _serialize_scalar(first_date),
        "last_usable_training_date": _serialize_scalar(last_date),
        "usable_training_row_count": int(usable.sum()),
        "usable_training_span_days": int((last_date - first_date).days + 1),
    }


def _target_diagnostics(df: pd.DataFrame, feature_coverage: dict[str, Any]) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    feature_rows = {
        row["commodity"]: row
        for row in feature_coverage.get("feature_non_null_rates_by_commodity", [])
    }

    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    for commodity, part in frame.groupby("commodity", sort=True):
        total = max(len(part), 1)
        part = part.sort_values("date").reset_index(drop=True)

        feature_info = feature_rows.get(commodity, {})
        critical_cols = [c for c in FEATURE_TIERS["CRITICAL"] if c in part.columns]
        model_mask = (
            part[critical_cols].notna().all(axis=1)
            if critical_cols
            else pd.Series([False] * len(part), index=part.index)
        )

        strict_mask = (
            part[list(FEATURE_COLUMNS)].notna().all(axis=1)
            if set(FEATURE_COLUMNS).issubset(part.columns)
            else pd.Series([False] * len(part), index=part.index)
        )
        
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
                "strict_features_and_target_7d_overlap_count": int((strict_mask & target_7d_valid).sum()),
                "strict_features_and_target_30d_overlap_count": int((strict_mask & target_30d_valid).sum()),
                "model_features_and_target_7d_overlap_count": int((model_mask & target_7d_valid).sum()),
                "model_features_and_target_30d_overlap_count": int((model_mask & target_30d_valid).sum()),
                "trainable_rows_7d": int((model_mask & target_7d_valid).sum()),
                "trainable_rows_30d": int((model_mask & target_30d_valid).sum()),
                "strict_feature_usable_rows": int(feature_info.get("strict_feature_usable_rows", 0)),
                "model_feature_usable_rows": int(feature_info.get("model_feature_usable_rows", 0)),
            }
        )

        window_7d = _training_window(part, model_mask, "target_7d_inflation_pct")
        window_30d = _training_window(part, model_mask, "target_30d_inflation_pct")
        window_rows.append(
            {
                "commodity": commodity,
                "horizon": "7d",
                **window_7d,
            }
        )
        window_rows.append(
            {
                "commodity": commodity,
                "horizon": "30d",
                **window_30d,
            }
        )

    table = pd.DataFrame(rows).sort_values("commodity") if rows else pd.DataFrame()
    windows = pd.DataFrame(window_rows).sort_values(["commodity", "horizon"]) if window_rows else pd.DataFrame()

    return {
        "by_commodity": _records(table),
        "training_windows": _records(windows),
        "trainable_rows_7d_total": int(table["trainable_rows_7d"].sum()) if not table.empty else 0,
        "trainable_rows_30d_total": int(table["trainable_rows_30d"].sum()) if not table.empty else 0,
    }

def _label_class(series: pd.Series, stable_band_pct: float) -> pd.Series:
    classes = pd.Series(index=series.index, dtype="object")
    classes.loc[series > stable_band_pct] = "Inflation"
    classes.loc[series < -stable_band_pct] = "Deflation"
    classes.loc[(series >= -stable_band_pct) & (series <= stable_band_pct)] = "Stable"
    return classes


def _class_counts_and_ratios(labels: pd.Series) -> tuple[dict[str, int], dict[str, float]]:
    counts = labels.value_counts(dropna=True)
    total = int(counts.sum())
    out_counts = {klass: int(counts.get(klass, 0)) for klass in CLASS_LABELS}
    out_ratios = {klass: float(out_counts[klass] / total) if total else 0.0 for klass in CLASS_LABELS}
    return out_counts, out_ratios


def _class_balance_diagnostics(
    df: pd.DataFrame,
    feature_coverage: dict[str, Any],
    config: DiagnosticsConfig,
) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    by_commodity: list[dict[str, Any]] = []
    overall: dict[str, Any] = {}

    horizon_columns = {"7d": "target_7d_inflation_pct", "30d": "target_30d_inflation_pct"}

    for horizon, target_col in horizon_columns.items():
        all_labels: list[pd.Series] = []
        for commodity, part in frame.groupby("commodity", sort=True):
            part = part.sort_values("date").reset_index(drop=True)
            critical_cols = [c for c in FEATURE_TIERS["CRITICAL"] if c in part.columns]
            model_mask = (
                part[critical_cols].notna().all(axis=1)
                if critical_cols
                else pd.Series([False] * len(part), index=part.index)
            )
            valid_mask = model_mask & part[target_col].notna()
            labels = _label_class(part.loc[valid_mask, target_col], config.stable_band_pct)
            all_labels.append(labels)
            counts, ratios = _class_counts_and_ratios(labels)
            by_commodity.append(
                {
                    "commodity": commodity,
                    "horizon": horizon,
                    "trainable_row_count": int(valid_mask.sum()),
                    "class_counts": counts,
                    "class_ratios": ratios,
                    "classes_present": int(sum(v > 0 for v in counts.values())),
                }
            )

        merged_labels = pd.concat(all_labels, ignore_index=True) if all_labels else pd.Series(dtype="object")
        counts, ratios = _class_counts_and_ratios(merged_labels)
        overall[horizon] = {
            "trainable_row_count": int(len(merged_labels)),
            "class_counts": counts,
            "class_ratios": ratios,
            "classes_present": int(sum(v > 0 for v in counts.values())),
        }

    return {
        "stable_band_pct": config.stable_band_pct,
        "overall": overall,
        "by_commodity": by_commodity,
    }


def _match_quality_by_commodity(
    part: pd.DataFrame,
    source_col: str,
    matched_date_col: str,
    target_date_col: str,
) -> pd.DataFrame:
    frame = part[["date", source_col, matched_date_col, target_date_col]].copy()
    frame[matched_date_col] = pd.to_datetime(frame[matched_date_col])
    frame[target_date_col] = pd.to_datetime(frame[target_date_col])

    no_match = frame[source_col].isna() | frame[matched_date_col].isna()
    exact = (~no_match) & (frame[matched_date_col] == frame[target_date_col])
    tolerance = (~no_match) & (~exact)
    frame["match_quality"] = "no_match"
    frame.loc[exact, "match_quality"] = "exact"
    frame.loc[tolerance, "match_quality"] = "tolerance"
    return frame


def _lag_target_match_quality_diagnostics(df: pd.DataFrame) -> dict[str, Any]:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows: list[dict[str, Any]] = []

    lag_specs = [
        ("price_lag_7d", DEFAULT_LAG_CONFIGS["price_lag_7d"].horizon_days, DEFAULT_LAG_CONFIGS["price_lag_7d"].max_lookback_days),
        ("price_lag_30d", DEFAULT_LAG_CONFIGS["price_lag_30d"].horizon_days, DEFAULT_LAG_CONFIGS["price_lag_30d"].max_lookback_days),
    ]
    target_specs = [
        (
            "target_7d_inflation_pct",
            DEFAULT_TARGET_CONFIGS["target_7d_inflation_pct"].horizon_days,
            DEFAULT_TARGET_CONFIGS["target_7d_inflation_pct"].max_lookahead_days,
        ),
        (
            "target_30d_inflation_pct",
            DEFAULT_TARGET_CONFIGS["target_30d_inflation_pct"].horizon_days,
            DEFAULT_TARGET_CONFIGS["target_30d_inflation_pct"].max_lookahead_days,
        ),
    ]

    for commodity, part in frame.groupby("commodity", sort=True):
        part = part.sort_values("date").reset_index(drop=True)

        for col_name, horizon_days, tol_days in lag_specs:
            if col_name not in part.columns:
                continue
            working = part[["date", col_name]].copy()
            working["target_date"] = working["date"] - pd.Timedelta(days=horizon_days)
            lookup = part[["date"]].rename(columns={"date": "matched_date"})
            matched = pd.merge_asof(
                working.sort_values("target_date"),
                lookup.sort_values("matched_date"),
                left_on="target_date",
                right_on="matched_date",
                direction="backward",
                tolerance=pd.Timedelta(days=tol_days),
            )
            quality_df = _match_quality_by_commodity(
                matched,
                source_col=col_name,
                matched_date_col="matched_date",
                target_date_col="target_date",
            )
            counts = quality_df["match_quality"].value_counts()
            total = int(len(quality_df))
            rows.append(
                {
                    "commodity": commodity,
                    "field": col_name,
                    "exact_count": int(counts.get("exact", 0)),
                    "tolerance_count": int(counts.get("tolerance", 0)),
                    "no_match_count": int(counts.get("no_match", 0)),
                    "exact_ratio": float(counts.get("exact", 0) / total) if total else 0.0,
                    "tolerance_ratio": float(counts.get("tolerance", 0) / total) if total else 0.0,
                    "no_match_ratio": float(counts.get("no_match", 0) / total) if total else 0.0,
                }
            )

        for col_name, horizon_days, tol_days in target_specs:
            if col_name not in part.columns:
                continue
            working = part[["date", col_name]].copy()
            working["target_date"] = working["date"] + pd.Timedelta(days=horizon_days)
            lookup = part[["date"]].rename(columns={"date": "matched_date"})
            matched = pd.merge_asof(
                working.sort_values("target_date"),
                lookup.sort_values("matched_date"),
                left_on="target_date",
                right_on="matched_date",
                direction="forward",
                tolerance=pd.Timedelta(days=tol_days),
            )
            quality_df = _match_quality_by_commodity(
                matched,
                source_col=col_name,
                matched_date_col="matched_date",
                target_date_col="target_date",
            )
            counts = quality_df["match_quality"].value_counts()
            total = int(len(quality_df))
            rows.append(
                {
                    "commodity": commodity,
                    "field": col_name,
                    "exact_count": int(counts.get("exact", 0)),
                    "tolerance_count": int(counts.get("tolerance", 0)),
                    "no_match_count": int(counts.get("no_match", 0)),
                    "exact_ratio": float(counts.get("exact", 0) / total) if total else 0.0,
                    "tolerance_ratio": float(counts.get("tolerance", 0) / total) if total else 0.0,
                    "no_match_ratio": float(counts.get("no_match", 0) / total) if total else 0.0,
                }
            )

    table = pd.DataFrame(rows).sort_values(["commodity", "field"]) if rows else pd.DataFrame()
    overall_rows = []
    if not table.empty:
        for field, part in table.groupby("field"):
            total_exact = int(part["exact_count"].sum())
            total_tol = int(part["tolerance_count"].sum())
            total_missing = int(part["no_match_count"].sum())
            total = total_exact + total_tol + total_missing
            overall_rows.append(
                {
                    "field": field,
                    "exact_count": total_exact,
                    "tolerance_count": total_tol,
                    "no_match_count": total_missing,
                    "exact_ratio": float(total_exact / total) if total else 0.0,
                    "tolerance_ratio": float(total_tol / total) if total else 0.0,
                    "no_match_ratio": float(total_missing / total) if total else 0.0,
                }
            )

    return {
        "by_commodity": _records(table),
        "overall": _records(pd.DataFrame(overall_rows).sort_values("field") if overall_rows else pd.DataFrame()),
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

def _max_flatline_segment_days(part: pd.DataFrame) -> int:
    series = pd.to_numeric(part["price_idr"], errors="coerce")
    if series.dropna().empty:
        return 0
    same_as_prev = series.eq(series.shift(1)).fillna(False)
    longest = current = 0
    for v in same_as_prev:
        if v:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest + 1 if longest else 1)


def _distribution_diagnostics(df: pd.DataFrame, config: DiagnosticsConfig) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for commodity, part in df.groupby("commodity", sort=True):
        part = part.sort_values("date").reset_index(drop=True)
        abs_1d = pd.to_numeric(part.get("price_change_1d_pct", pd.Series(dtype=float)), errors="coerce").abs().dropna()
        abs_7d = pd.to_numeric(part.get("price_change_7d_pct", pd.Series(dtype=float)), errors="coerce").abs().dropna()
        abs_30d = pd.to_numeric(part.get("price_change_30d_pct", pd.Series(dtype=float)), errors="coerce").abs().dropna()

        def max_and_ratio(clean: pd.Series) -> tuple[float | None, float | None]:
            if clean.empty:
                return None, None
            median = float(clean.median())
            max_v = float(clean.max())
            if median <= 0:
                return max_v, None
            return max_v, float(max_v / median)

        max_1d, ratio_1d = max_and_ratio(abs_1d)
        max_7d, ratio_7d = max_and_ratio(abs_7d)
        max_30d, ratio_30d = max_and_ratio(abs_30d)
        flatline_days = _max_flatline_segment_days(part)
        structural_jump_count = int((abs_1d > config.thresholds.structural_jump_ratio_warn).sum()) if not abs_1d.empty else 0

        rec = {
            "commodity": commodity,
            "price_idr": _summary_stats(part["price_idr"]),
            "price_change_1d_pct": _summary_stats(part.get("price_change_1d_pct", pd.Series(dtype=float))),
            "price_change_7d_pct": _summary_stats(part.get("price_change_7d_pct", pd.Series(dtype=float))),
            "price_change_30d_pct": _summary_stats(part.get("price_change_30d_pct", pd.Series(dtype=float))),
            "target_7d_inflation_pct": _summary_stats(part.get("target_7d_inflation_pct", pd.Series(dtype=float))),
            "target_30d_inflation_pct": _summary_stats(part.get("target_30d_inflation_pct", pd.Series(dtype=float))),
            "max_abs_price_change_1d_pct": max_1d,
            "max_abs_price_change_7d_pct": max_7d,
            "max_abs_price_change_30d_pct": max_30d,
            "max_to_median_abs_change_ratio_1d": ratio_1d,
            "max_to_median_abs_change_ratio_7d": ratio_7d,
            "max_to_median_abs_change_ratio_30d": ratio_30d,
            "max_flatline_segment_days": flatline_days,
            "suspicious_structural_jump_count": structural_jump_count,
        }
        rows.append(rec)

    return {"by_commodity": rows}


def evaluate_readiness(
    canonical: dict[str, Any],
    gaps: dict[str, Any],
    feature_coverage: dict[str, Any],
    target_coverage: dict[str, Any],
    class_balance: dict[str, Any],
    lag_target_match_quality: dict[str, Any],
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
        commodity = row["commodity"]
        for critical_col in FEATURE_TIERS["CRITICAL"]:
            rate = float(row.get(f"{critical_col}_non_null_rate", 0.0))
            if rate < t.min_critical_feature_non_null_rate_fail:
                failures.append(
                    f"{commodity}: CRITICAL feature {critical_col} non-null rate too low ({rate:.3f})."
                )
            elif rate < t.min_critical_feature_non_null_rate_warn:
                warnings.append(
                    f"{commodity}: CRITICAL feature {critical_col} non-null rate is weak ({rate:.3f})."
                )

        model_rate = float(row.get("model_feature_usable_rate", 0.0))
        if model_rate < t.min_model_feature_usable_rate_warn:
            warnings.append(
                f"{commodity}: model-feature usable rate is low ({model_rate:.3f})."
            )

    for row in target_coverage.get("by_commodity", []):
        commodity = row["commodity"]
        for horizon_col in TARGET_COLUMNS:
            rate_key = f"{horizon_col.replace('_inflation_pct', '')}_non_null_rate"
            rate = float(row.get(rate_key, 0.0))
            if rate < t.min_target_non_null_rate_warn:
                warnings.append(f"{commodity}: low non-null rate for {horizon_col} ({rate:.3f}).")
            if rate < t.min_target_non_null_rate_per_commodity:
                failures.append(
                    f"{commodity}: target coverage below per-commodity minimum for {horizon_col} ({rate:.3f})."
                )

        if int(row.get("trainable_rows_7d", 0)) < t.min_trainable_rows_7d_per_commodity:
            failures.append(f"{commodity}: insufficient trainable rows for 7d horizon.")
        if int(row.get("trainable_rows_30d", 0)) < t.min_trainable_rows_30d_per_commodity:
            failures.append(f"{commodity}: insufficient trainable rows for 30d horizon.")

    if target_coverage.get("trainable_rows_7d_total", 0) < t.min_trainable_rows_7d:
        failures.append("Insufficient total trainable rows for 7d horizon.")
    if target_coverage.get("trainable_rows_30d_total", 0) < t.min_trainable_rows_30d:
        failures.append("Insufficient total trainable rows for 30d horizon.")

    for row in target_coverage.get("training_windows", []):
        span = int(row.get("usable_training_span_days", 0))
        commodity = row["commodity"]
        horizon = row["horizon"]
        if horizon == "7d" and span < t.min_training_span_days_7d_per_commodity:
            failures.append(f"{commodity}: usable training span too short for 7d horizon ({span} days).")
        if horizon == "30d" and span < t.min_training_span_days_30d_per_commodity:
            failures.append(f"{commodity}: usable training span too short for 30d horizon ({span} days).")

    for row in class_balance.get("by_commodity", []):
        commodity = row["commodity"]
        horizon = row["horizon"]
        class_ratios: dict[str, float] = row.get("class_ratios", {})
        classes_present = int(row.get("classes_present", 0))
        if classes_present < t.min_classes_present_per_commodity:
            failures.append(f"{commodity}: class support too narrow for {horizon} ({classes_present} classes).")
        for klass, ratio in class_ratios.items():
            if ratio < t.min_class_ratio_fail:
                failures.append(f"{commodity}: class {klass} too rare for {horizon} ({ratio:.3f}).")
            elif ratio < t.min_class_ratio_warn:
                warnings.append(f"{commodity}: class {klass} imbalanced for {horizon} ({ratio:.3f}).")

    for row in lag_target_match_quality.get("overall", []):
        field = row["field"]
        tolerance_ratio = float(row.get("tolerance_ratio", 0.0))
        no_match_ratio = float(row.get("no_match_ratio", 0.0))
        if tolerance_ratio > t.max_tolerance_match_ratio_fail:
            failures.append(
                f"{field}: tolerance-based matching ratio too high ({tolerance_ratio:.3f})."
            )
        elif tolerance_ratio > t.max_tolerance_match_ratio_warn:
            warnings.append(
                f"{field}: tolerance-based matching ratio elevated ({tolerance_ratio:.3f})."
            )

        if no_match_ratio > (1.0 - t.min_target_non_null_rate_per_commodity):
            warnings.append(f"{field}: no-match ratio is high ({no_match_ratio:.3f}).")

    for row in distributions.get("by_commodity", []):
        commodity = row["commodity"]
        p90_abs_change = abs((row.get("price_change_1d_pct") or {}).get("p90") or 0.0)
        if p90_abs_change > t.max_abs_price_change_pct_fail:
            failures.append(f"{commodity}: extreme p90 1d price change ({p90_abs_change:.2f}%).")
        elif p90_abs_change > t.max_abs_price_change_pct_warn:
            warnings.append(f"{commodity}: elevated p90 1d price change ({p90_abs_change:.2f}%).")

        for ratio_key in (
            "max_to_median_abs_change_ratio_1d",
            "max_to_median_abs_change_ratio_7d",
            "max_to_median_abs_change_ratio_30d",
        ):
            ratio = row.get(ratio_key)
            if ratio is None:
                continue
            ratio = float(ratio)
            if ratio > t.max_change_to_median_ratio_fail:
                failures.append(f"{commodity}: {ratio_key} too high ({ratio:.2f}).")
            elif ratio > t.max_change_to_median_ratio_warn:
                warnings.append(f"{commodity}: {ratio_key} elevated ({ratio:.2f}).")

        flatline = int(row.get("max_flatline_segment_days", 0))
        if flatline > t.max_flatline_segment_days_fail:
            failures.append(f"{commodity}: prolonged flatline segment ({flatline} days).")
        elif flatline > t.max_flatline_segment_days_warn:
            warnings.append(f"{commodity}: flatline segment is long ({flatline} days).")

        jumps = int(row.get("suspicious_structural_jump_count", 0))
        if jumps > 0 and (row.get("max_to_median_abs_change_ratio_1d") or 0) > t.structural_jump_ratio_fail:
            failures.append(f"{commodity}: suspicious structural jumps detected ({jumps} rows).")
        elif jumps > 0 and (row.get("max_to_median_abs_change_ratio_1d") or 0) > t.structural_jump_ratio_warn:
            warnings.append(f"{commodity}: potential structural jumps detected ({jumps} rows).")

    if failures:
        return ReadinessStatus.FAIL, sorted(set(failures)), sorted(set(warnings))
    if warnings:
        return ReadinessStatus.PASS_WITH_WARNINGS, [], sorted(set(warnings))
    return ReadinessStatus.PASS, [], []
