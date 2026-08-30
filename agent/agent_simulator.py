"""Model-driven policy simulator for the Revenue Agent.

Uses the causal engine's per-row predictions (baseline and treated conversion
probabilities, contribution margin, canonical discount) to estimate expected
orders, revenue, discount cost, and profit under a given discount policy.

One source of truth: everything is derived from the same predictions frame that
drives segmentation and leakage, so numbers reconcile across pages.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from .agent_tools import rows_from_predictions

DEFAULT_POLICY = {
    "sure_thing_discount_cap": .25,
    "persuadable_max": .20,
    "price_sensitive_max": .25,
}

SEGMENT_KEY = {
    "Sure Thing": "sure_thing_discount_cap",
    "Persuadable": "persuadable_max",
    "Price Sensitive": "price_sensitive_max",
    "Lost Cause": "sure_thing_discount_cap",
}


def _cap_for(policy: Dict[str, Any], segment: str) -> float:
    key = SEGMENT_KEY.get(segment, "sure_thing_discount_cap")
    return float(policy.get(key, DEFAULT_POLICY.get(key, 0.0)))


def _discount_fraction(row: pd.Series) -> float:
    value = float(row.get("original_value") or row.get("final_value") or 0.0)
    discount = float(row.get("discount_amount_canonical") or 0.0)
    return discount / value if value > 0 else 0.0


def _project(rows: pd.DataFrame, policy: Dict[str, Any]) -> Dict[str, float]:
    value = pd.to_numeric(rows.get("original_value", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0).clip(lower=0)
    margin = pd.to_numeric(rows.get("contribution_margin", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0).clip(lower=0)
    p0 = pd.to_numeric(rows.get("baseline_probability", pd.Series(0.0, index=rows.index)), errors="coerce").fillna(0).clip(0, 1)
    p1 = pd.to_numeric(rows.get("treated_probability", pd.Series(p0.values, index=rows.index)), errors="coerce").fillna(p0).clip(0, 1)
    segment = rows.get("customer_type", pd.Series("Price Sensitive", index=rows.index)).fillna("Price Sensitive")

    observed_discount = value.copy()
    if "discount_amount_canonical" in rows.columns:
        discount_amount = pd.to_numeric(rows["discount_amount_canonical"], errors="coerce").fillna(0).clip(lower=0)
        observed_discount = discount_amount / value.where(value > 0, np.nan)
        observed_discount = observed_discount.fillna(0)

    capped = observed_discount.copy()
    for seg in SEGMENT_KEY:
        mask = segment == seg
        cap = _cap_for(policy, seg)
        capped = capped.where(~mask, capped.clip(upper=cap).where(mask, capped))

    # Per-unit treatment effect slope: (p1 - p0) / observed_discount.
    slope = (p1 - p0) / observed_discount.where(observed_discount > 0, np.nan)
    slope = slope.fillna(0)

    probability = (p0 + slope * capped).clip(0, 1)
    discount_paid = value * capped

    orders = float(probability.sum())
    revenue = float((probability * value).sum())
    discount_cost = float((probability * discount_paid).sum())
    profit = float((probability * (margin - discount_paid)).sum())
    return {"orders": round(orders, 1), "revenue": round(revenue, 2), "discount_cost": round(discount_cost, 2), "profit": round(profit, 2)}


def simulate_policy(current_policy: Dict[str, Any], proposed_policy: Dict[str, Any], state: Dict[str, Any]) -> Dict[str, Any]:
    rows = rows_from_predictions(state.get("model", {}))
    if rows.empty:
        return {"error": "No model predictions available for simulation."}
    current = _project(rows, current_policy or DEFAULT_POLICY)
    proposed = _project(rows, proposed_policy or DEFAULT_POLICY)
    deltas = {
        "orders_delta": round(proposed["orders"] - current["orders"], 1),
        "revenue_delta": round(proposed["revenue"] - current["revenue"], 2),
        "discount_delta": round(proposed["discount_cost"] - current["discount_cost"], 2),
        "profit_delta": round(proposed["profit"] - current["profit"], 2),
        "margin_recovered": round(current["discount_cost"] - proposed["discount_cost"], 2),
        "revenue_at_risk": round(current["revenue"] - proposed["revenue"], 2),
    }
    return {
        "current": current,
        "proposed": proposed,
        "deltas": deltas,
        "policy_applied": proposed_policy,
        "reliability": state.get("model", {}).get("reliability", "LOW"),
        "profit_is_proxy": state.get("model", {}).get("profit_is_proxy", True),
    }
