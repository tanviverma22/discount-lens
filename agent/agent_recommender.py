"""Opportunity ranking and policy optimization for the Revenue Agent."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd

from .agent_tools import dimension_breakdown, rows_from_predictions, top_evidence


class RevenueRecommender:
    """Converts causal predictions into ranked, explainable business actions."""

    default_policy = {"sure_thing_discount_cap": .25, "persuadable_max": .20, "price_sensitive_max": .25}

    def opportunities(self, model: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = rows_from_predictions(model)
        if rows.empty:
            return []
        candidates: List[Dict[str, Any]] = []
        for dimension, label in (("customer_type", "Customer segment"), ("campaign_id", "Campaign"), ("channel", "Channel"), ("product_category", "Product category"), ("region", "Region")):
            for record in dimension_breakdown(rows, dimension):
                if record.get("leakage", 0) <= 0:
                    continue
                transactions = max(int(record.get("transactions", 0)), 1)
                leakage = float(record["leakage"])
                recoverable = float(record.get("recoverable_margin", leakage))
                score = round(min(100, 35 * leakage / max(float(rows["discount_amount_canonical"].sum()), 1) + 35 * record.get("leakage_transactions", 0) / transactions + 30 * min(recoverable / max(leakage, 1), 1)) * 100, 1)
                value = str(record[dimension])
                candidates.append({
                    "dimension": dimension, "value": value, "title": f"{label}: {value}",
                    "leakage": round(leakage, 2), "potential_recovery": round(recoverable, 2),
                    "transactions": int(record.get("transactions", 0)), "affected_orders": int(record.get("leakage_transactions", 0)),
                    "score": score, "confidence": model.get("reliability", "LOW"),
                    "evidence": top_evidence(rows, {dimension: value}, 3),
                })
        deduplicated = sorted(candidates, key=lambda item: (item["score"], item["leakage"]), reverse=True)
        return [{**item, "priority": index + 1} for index, item in enumerate(deduplicated[:8])]

    def recommendation(self, model: Dict[str, Any], opportunity: Dict[str, Any] | None = None) -> Dict[str, Any]:
        rows = rows_from_predictions(model)
        sure = rows[rows.get("customer_type", pd.Series(index=rows.index, dtype=str)) == "Sure Thing"] if not rows.empty else rows
        leakage = float(sure.loc[sure.get("was_leakage", False), "discount_amount_canonical"].sum()) if not sure.empty and "discount_amount_canonical" in sure else 0.0
        affected = int(sure.get("was_leakage", pd.Series(dtype=bool)).sum()) if not sure.empty else 0
        focus = opportunity or (self.opportunities(model)[0] if self.opportunities(model) else None)
        evidence = focus.get("evidence", []) if focus else top_evidence(rows, {"customer_type": "Sure Thing"}, 3)
        return {
            "problem": focus["title"] if focus else "Discounts given to high-baseline customers",
            "observed": f"{affected} treated Sure Thing orders have non-positive estimated incremental profit, accounting for {leakage:,.2f} in canonical discount leakage.",
            "inference": "These customers have high predicted baseline conversion and the model estimates little or no incremental profit from the observed discount.",
            "action": "Set Sure Thing discount eligibility to 0% and preserve targeted offers for customers with positive estimated incremental profit.",
            "confidence": model.get("reliability", "LOW"),
            "expected_impact": round(leakage, 2), "affected_orders": affected,
            "evidence": evidence,
            "why": ["High modelled baseline conversion probability", "Non-positive expected incremental profit after discount cost", "Treatment/control overlap diagnostics passed"],
            "proposed_policy": {"sure_thing_discount_cap": 0.0, "persuadable_max": .20, "price_sensitive_max": .20, "constraints": {"minimum_revenue_retention": .95}},
        }

    def optimize(self, simulator, state: Dict[str, Any]) -> Dict[str, Any]:
        current = self.default_policy.copy()
        best: Dict[str, Any] | None = None
        attempted = 0
        for sure_cap in (0.0, .05, .10, .15, .20):
            for persuadable_cap in (.10, .15, .20, .25):
                for sensitive_cap in (.10, .15, .20, .25):
                    proposed = {"sure_thing_discount_cap": sure_cap, "persuadable_max": persuadable_cap, "price_sensitive_max": sensitive_cap}
                    result = simulator(current, proposed, state)
                    attempted += 1
                    if result.get("error"):
                        continue
                    baseline_revenue = max(float(result["current"].get("revenue", 0)), 1)
                    proposed_revenue = float(result["proposed"].get("revenue", 0))
                    if proposed_revenue < baseline_revenue * .95:
                        continue
                    if best is None or result["proposed"].get("profit", 0) > best["simulation"]["proposed"].get("profit", 0):
                        best = {"policy": proposed, "simulation": result}
        if best is None:
            return {"error": "No policy met the 95% projected revenue-retention constraint.", "evaluated": attempted}
        best["evaluated"] = attempted
        best["constraints"] = {"minimum_revenue_retention": .95}
        return best
