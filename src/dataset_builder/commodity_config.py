from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommoditySpec:
    """Canonical commodity specification for PIHPS top-level rows."""

    raw_label: str
    canonical_label: str


# v1 scope: national/all-region top-level commodities only.
TOP_LEVEL_COMMODITY_SPECS: tuple[CommoditySpec, ...] = (
    CommoditySpec(raw_label="Beras", canonical_label="beras"),
    CommoditySpec(raw_label="Cabai Merah", canonical_label="cabai_merah"),
    CommoditySpec(raw_label="Bawang Merah", canonical_label="bawang_merah"),
    CommoditySpec(raw_label="Telur Ayam", canonical_label="telur_ayam"),
    CommoditySpec(raw_label="Minyak Goreng", canonical_label="minyak_goreng"),
    CommoditySpec(raw_label="Daging Ayam", canonical_label="daging_ayam"),
    CommoditySpec(raw_label="Gula Pasir", canonical_label="gula_pasir"),
)

RAW_TO_CANONICAL: dict[str, str] = {
    spec.raw_label: spec.canonical_label for spec in TOP_LEVEL_COMMODITY_SPECS
}
ALLOWED_RAW_COMMODITIES: set[str] = set(RAW_TO_CANONICAL)
ALLOWED_CANONICAL_COMMODITIES: set[str] = set(RAW_TO_CANONICAL.values())
