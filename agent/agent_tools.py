"""Data tools used by the Revenue Agent."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import pandas as pd


def rows_from_predictions(model: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(model.get("predictions", []))


def dimension_breakdown(rows: pd.DataFrame, dimension: str) -> List[Dict[str, Any]]:
    """Aggregate canonical leakage and financial evidence by an available dimension."""
    if rows.empty or dimension not in rows.columns:
        return []
    work = rows.copy()
    for column in ("discount_amount_canonical", "expected_incremental_profit", "expected_incremental_revenue"):
        if column not in work:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0)
    if "was_leakage" not in work:
        work["was_leakage"] = False
    grouped = work.groupby(dimension, dropna=False).agg(
        transactions=(dimension, "size"),
        treated=("treated", "sum"),
        leakage_transactions=("was_leakage", "sum"),
        discount_spend=("discount_amount_canonical", "sum"),
        leakage=("discount_amount_canonical", lambda values: float(values[work.loc[values.index, "was_leakage"]].sum())),
        expected_incremental_profit=("expected_incremental_profit", "sum"),
    ).reset_index()
    grouped["recoverable_margin"] = grouped["leakage"]
    records = grouped.sort_values("leakage", ascending=False).to_dict("records")
    return [{key: (value.item() if hasattr(value, "item") else value) for key, value in record.items()} for record in records]


def top_evidence(rows: pd.DataFrame, filters: Dict[str, Any] | None = None, limit: int = 5) -> List[Dict[str, Any]]:
    if rows.empty:
        return []
    work = rows.copy()
    for column, value in (filters or {}).items():
        if column in work.columns:
            work = work[work[column].astype(str).str.lower() == str(value).lower()]
    if "was_leakage" in work.columns:
        work = work[work["was_leakage"]]
    if "discount_amount_canonical" in work.columns:
        work = work.sort_values("discount_amount_canonical", ascending=False)
    fields = [field for field in ("transaction_id", "customer_id", "customer_type", "channel", "campaign_id", "product_category", "region", "treated", "conversion", "original_value", "discount_amount_canonical", "baseline_probability", "treated_probability", "estimated_incremental_lift", "expected_incremental_profit") if field in work.columns]
    return work[fields].head(limit).to_dict("records")


def policy_json(policy: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "policy_version": "1.0",
        "objective": "maximize_expected_profit",
        "rules": [
            {"segment": "sure_thing", "max_discount": policy.get("sure_thing_discount_cap", 0)},
            {"segment": "persuadable", "max_discount": policy.get("persuadable_max", 0)},
            {"segment": "price_sensitive", "max_discount": policy.get("price_sensitive_max", policy.get("price_warrior_max", 0))},
        ],
        "constraints": policy.get("constraints", {"minimum_revenue_retention": .95}),
    }
