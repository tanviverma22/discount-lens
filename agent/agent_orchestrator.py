"""Tool-driven Revenue Agent orchestration."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from .agent_memory import AgentMemory
from .agent_recommender import RevenueRecommender
from .agent_tools import dimension_breakdown, rows_from_predictions, top_evidence


class RevenueAgent:
    """Coordinates data-derived observation, investigation, recommendation, and action."""

    def __init__(self, memory: AgentMemory) -> None:
        self.memory = memory
        self.recommender = RevenueRecommender()

    def build_state(self, dataset_name: str, schema: Dict[str, Any], validation: Dict[str, Any], model: Dict[str, Any], descriptive: Dict[str, Any]) -> Dict[str, Any]:
        state = {
            "dataset": {"name": dataset_name, "rows": int(schema.get("row_count", 0)), "schema": schema},
            "validation": validation,
            "model": model,
            "descriptive": descriptive,
            "mode": "causal" if model.get("available") else "descriptive",
            "created_at": datetime.utcnow().isoformat() + "Z",
        }
        state["opportunities"] = self.recommender.opportunities(model) if model.get("available") else []
        state["recommendation"] = self.recommender.recommendation(model) if model.get("available") else None
        state["brief"] = self._brief(state)
        state["trail"] = self._trail(state)
        return state

    def _trail(self, state: Dict[str, Any]) -> List[Dict[str, str]]:
        dataset = state["dataset"]
        model = state["model"]
        validation = state["validation"]
        trail = [
            {"step": "OBSERVE", "message": f"Inspected {dataset['rows']:,} records and {len(dataset['schema'].get('columns', []))} columns."},
            {"step": "VALIDATE", "message": f"Mapped {len(dataset['schema'].get('mappings', {}))} semantic fields and ran data-quality checks."},
        ]
        if model.get("available"):
            diag = model["diagnostics"]
            trail.extend([
                {"step": "MODEL", "message": f"Trained propensity and treatment/control outcome models on {diag['train_rows']:,} training records; held out {diag['test_rows']:,} records."},
                {"step": "CHECK", "message": f"Common support covers {diag['common_support']['share'] * 100:.1f}% of observations ({model['reliability'].lower()} reliability)."},
                {"step": "INVESTIGATE", "message": f"Generated uplift and expected-profit estimates for {len(model.get('predictions', [])):,} observations."},
                {"step": "RECOMMEND", "message": f"Ranked {len(state['opportunities'])} revenue-recovery opportunities from canonical leakage."},
            ])
        else:
            reasons = [item["message"] for item in validation.get("findings", []) if item["level"] == "error"]
            trail.append({"step": "LIMIT", "message": reasons[0] if reasons else "Causal estimation is unavailable for this dataset."})
        return trail

    def _brief(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not state["model"].get("available"):
            return {
                "title": "Revenue Agent Brief", "status": "analysis limited", "issues": [],
                "summary": "Descriptive analysis is available, but this dataset lacks the treatment/control outcomes or pre-treatment features required for reliable causal estimates.",
                "next_step": "Upload conversion outcomes and pre-treatment behavioral data, or load the causal demo dataset.",
                "reliability": "INSUFFICIENT",
            }
        rec = state.get("recommendation") or {}
        opps = state.get("opportunities", [])
        return {
            "title": "Daily Revenue Intelligence Brief", "status": "opportunity detected" if opps else "no material opportunity detected",
            "issues": opps[:3],
            "summary": rec.get("observed", "No recoverable discount leakage was identified."),
            "next_step": rec.get("action", "Continue monitoring treatment response."),
            "potential_recovery": rec.get("expected_impact", 0),
            "reliability": state["model"].get("reliability", "LOW"),
        }

    def investigate(self, session_id: str, question: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Select data tools from request terms and active context; never use fixed answers."""
        query = (question or "").lower().strip()
        rows = rows_from_predictions(state.get("model", {}))
        if rows.empty:
            return self._limited_answer(state)

        dimension = self._dimension(query, rows)
        focus = self._find_value(query, rows, dimension)
        memory = self.memory.get(session_id)
        # If no dimension in this question, use the prior focus for continuity.
        if not dimension and memory.get("focus"):
            dimension, focus = memory["focus"].get("dimension"), memory["focus"].get("value")
        # If a dimension was detected but no specific value was named, anchor
        # on the highest-leakage value so "which channel leaks the most?" still
        # sets context for the follow-up "what should we do about it?".
        if dimension and not focus and dimension in rows:
            breakdown = dimension_breakdown(rows, dimension)
            if breakdown:
                focus = str(breakdown[0][dimension])
        if dimension and focus:
            self.memory.set_focus(session_id, dimension, str(focus))

        filtered = rows
        if dimension and focus and dimension in rows:
            filtered = rows[rows[dimension].astype(str).str.lower() == str(focus).lower()]
        leakage = float(filtered.loc[filtered["was_leakage"], "discount_amount_canonical"].sum()) if "was_leakage" in filtered else 0.0
        affected = int(filtered.get("was_leakage", []).sum()) if "was_leakage" in filtered else 0
        avg_baseline = float(filtered.get("baseline_probability", []).mean()) if len(filtered) else 0.0
        avg_lift = float(filtered.get("estimated_incremental_lift", []).mean()) if len(filtered) else 0.0
        evidence = top_evidence(rows, {dimension: focus} if dimension and focus else None, 5)
        question_kind = self._intent(query)
        label = f"{dimension.replace('_', ' ').title()}: {focus}" if dimension and focus else "the active dataset"
        recommendation = state.get("recommendation") or self.recommender.recommendation(state["model"])
        response = {
            "title": f"Revenue Agent — {question_kind.title()}",
            "finding": f"{label} contains {affected:,} treated observations with non-positive estimated incremental profit and {leakage:,.2f} in canonical discount leakage.",
            "observed": {"scope": label, "rows": int(len(filtered)), "leakage": round(leakage, 2), "affected_orders": affected, "average_baseline_probability": round(avg_baseline, 4), "average_incremental_lift": round(avg_lift, 4)},
            "inference": "The model estimates these discounts are unlikely to create enough incremental conversion value to offset their cost. This is an observational estimate, not proof of individual intent.",
            "recommendation": recommendation,
            "evidence": evidence,
            "confidence": state["model"].get("reliability", "LOW"),
            "context": self.memory.get(session_id).get("focus", {}),
            "tools_used": ["analyze_segments", "calculate_discount_leakage", "get_transaction_evidence"] + ([f"analyze_{dimension}"] if dimension else []),
        }
        return response

    @staticmethod
    def _intent(query: str) -> str:
        if any(word in query for word in ("why", "explain", "evidence")):
            return "explanation"
        if any(word in query for word in ("what should", "fix", "next", "action", "recommend")):
            return "recommendation"
        if any(word in query for word in ("simulate", "happen", "stop")):
            return "simulation"
        return "investigation"

    @staticmethod
    def _dimension(query: str, rows) -> Optional[str]:
        candidates = {"campaign": "campaign_id", "channel": "channel", "product": "product_category", "category": "product_category", "region": "region", "segment": "customer_type", "customer": "customer_id", "sure thing": "customer_type", "persuadable": "customer_type", "lost cause": "customer_type"}
        for phrase, column in candidates.items():
            if phrase in query and column in rows.columns:
                return column
        return None

    @staticmethod
    def _find_value(query: str, rows, dimension: Optional[str]) -> Optional[str]:
        if dimension is None or dimension not in rows:
            return None
        for value in rows[dimension].dropna().astype(str).unique():
            if value.lower() in query:
                return value
        for segment in ("Sure Thing", "Persuadable", "Lost Cause", "Price Sensitive"):
            if segment.lower() in query and dimension == "customer_type":
                return segment
        return None

    @staticmethod
    def _limited_answer(state: Dict[str, Any]) -> Dict[str, Any]:
        findings = state.get("validation", {}).get("findings", [])
        return {"title": "Revenue Agent — Analysis unavailable", "finding": "Reliable causal/uplift estimates are unavailable for the active dataset.", "observed": {"validation": findings}, "inference": "Without treatment/control conversion outcomes, any claim about discounts causing purchases would be speculative.", "recommendation": {"action": "Load the causal demo dataset or add treatment, conversion, and pre-treatment behavioral fields."}, "evidence": [], "confidence": "INSUFFICIENT", "tools_used": ["inspect_dataset", "validate_dataset"]}
