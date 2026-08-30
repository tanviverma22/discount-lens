"""
Schema-agnostic detection and resolution engine.

Inspects an arbitrary uploaded CSV and determines:
  * which semantic concepts are present (customer, treatment, conversion, money, profit, behavior)
  * how discount information is represented (one of 8 supported cases)
  * which analyses the data can support (capability matrix)
  * an agent-readable explanation of what was found

It NEVER fabricates missing values. Derived fields (discount_amount, discount_percent,
final_value, original_value) are computed ONLY when a defensible derivation exists from
present columns, and every derivation is recorded so the UI can label it.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np


# ---- semantic concepts -> match hints ----------------------------------------
# Each concept lists keyword fragments. A column matches if any fragment appears in
# its normalized name (lowercased, non-alnum -> underscore). Exact/contains matches
# rank higher than substring keyword matches.
SEMANTIC_HINTS: Dict[str, List[str]] = {
    "customer_id": ["customer", "user", "buyer", "client", "account", "shopper", "visitor"],
    "transaction_id": ["transaction", "transaction_id", "txn", "invoice", "receipt",
                       "purchase_id", "order_id", "order_number", "order_num"],
    "conversion": ["purchase_outcome", "converted", "conversion", "purchased", "bought",
                   "purchase", "is_purchase", "order_placed", "completed", "outcome",
                   "order_status", "order", "status", "order_flag", "purchase_flag"],
    "treatment": ["coupon_used", "discount_offered", "promotion_flag", "treated",
                  "is_discounted", "had_discount", "received_discount", "treatment",
                  "discount_applied", "used_coupon", "promo_used", "is_treated",
                  "discount_flag", "coupon", "promo"],
    "coupon_code": ["coupon_code", "coupon", "promo_code", "voucher", "discount_code",
                    "code", "promo"],
    "discount_amount": ["discount_amount", "discount_value", "savings", "reduction",
                        "discount", "amount_off", "discount_given"],
    "discount_percent": ["discount_percent", "discount_pct", "discount_rate", "percent_off",
                         "pct_off", "discount_percentage", "off_pct", "discount_rate"],
    "original_value": ["original_price", "base_value", "base_price", "list_price",
                       "original_value", "gross", "subtotal", "mrp", "price_before",
                       "pre_discount", "msrp", "full_price", "regular_price", "base"],
    "final_value": ["final_price", "final_value", "net", "paid", "revenue", "order_value",
                    "amount_paid", "net_price", "sale_price", "total_paid", "final",
                    "order_total", "net_value", "amount", "selling_price", "paid_price",
                    "transaction_value", "discounted_price", "price_paid", "price_after",
                    "charged", "actual_price", "payable"],
    "profit": ["profit", "margin", "net_profit", "gross_profit", "contribution",
               "net_margin", "gross_margin", "profit_margin"],
    "revenue": ["revenue", "sales", "gross_sales", "topline"],
    # behavioral signals
    "dwell_time": ["dwell", "time_on_page", "session_duration", "time_spent", "duration",
                   "checkout_time", "time_in_checkout"],
    "product_views": ["product_view", "views", "page_views", "items_viewed", "view_count"],
    "price_comparison": ["price_comparison", "comparison_tab", "compare", "price_compare",
                         "tabs_open", "comparison"],
    "cart_added": ["cart_added", "added_to_cart", "in_cart", "cart_add", "has_cart"],
    "days_since_last_visit": ["days_since", "last_visit", "recency", "days_last"],
    "session_id": ["session", "session_id"],
    "campaign_id": ["campaign", "promotion", "program", "campaign_id"],
    "channel": ["channel", "source", "medium", "traffic_source"],
    "timestamp": ["timestamp", "date", "time", "purchase_date", "created"],
    "product_category": ["category", "product_category", "cat", "dept", "department"],
}


def _normalize(name: str) -> str:
    s = str(name).lower().strip()
    out = []
    for ch in s:
        out.append(ch if ch.isalnum() else "_")
    n = "".join(out)
    while "__" in n:
        n = n.replace("__", "_")
    return n.strip("_")


def _classify_column(col: str, dtype: str) -> Optional[str]:
    """Return the best-matching semantic concept for a column, or None."""
    raw = str(col).strip()
    norm = _normalize(raw)

    # A percent/rate marker on a discount column means percentage, not amount.
    # e.g. "Discount %" / "discount_pct" / "discount_rate" -> discount_percent,
    # while "discount_amount" / "discount_value" -> discount_amount.
    pct_marker = any(tok in raw.lower() for tok in ("%", "pct", "rate", "percent", "off"))
    amt_marker = any(tok in norm for tok in ("amount", "value", "savings", "reduction"))

    best_concept, best_score = None, 0
    for concept, hints in SEMANTIC_HINTS.items():
        for hint in hints:
            score = 0
            if norm == hint:
                score = 100
            elif norm == hint.replace("_", ""):
                score = 96
            elif hint in norm:
                score = 80
            elif hint.replace("_", "") in norm:
                score = 70
            if score > best_score:
                best_concept, best_score = concept, score

    # Disambiguate discount amount vs percent when the column is a bare "discount".
    if best_concept == "discount_amount" and pct_marker and not amt_marker:
        best_concept = "discount_percent"

    # Disambiguate behavioral columns from the conversion outcome. Columns like
    # "prior_orders", "product_views", "previous_orders" contain "order"/"purchase"
    # tokens and would otherwise be misclassified as conversion, overwriting the
    # real outcome column. Reserve conversion for a genuine binary/flag-style name.
    if best_concept == "conversion":
        for token in ("prior", "previous", "views", "count", "history"):
            if token in norm:
                best_concept, best_score = None, 0
                break

    # require a minimum confidence to avoid spurious matches on generic columns
    if best_score >= 65:
        return best_concept, int(best_score)
    return None, int(best_score)


def detect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Inspect a dataframe and return a schema-detection result:
      mappings:     {concept: {column, dtype, confidence, sample}}
      derivations:  list of {field, from, formula} for derived canonical fields
      discount_case: one of 1..8 (str label) or None
      capabilities: {analysis: 'READY'|'LIMITED'|'NEEDS_DATA', reason}
      agent_reasoning: list[str] plain-English steps
      missing_concepts: concepts not found
    """
    columns = list(df.columns)
    dtypes = {c: str(df[c].dtype) for c in columns}
    mappings: Dict[str, Dict[str, Any]] = {}
    for col in columns:
        concept, score = _classify_column(col, dtypes[col])
        if concept and concept not in mappings:
            sample = _safe_sample(df[col])
            mappings[concept] = {
                "column": col,
                "dtype": dtypes[col],
                "confidence": score,
                "sample": sample,
            }

    # ---- resolve discount representation (8 cases) ---------------------------
    has = lambda c: c in mappings  # noqa: E731
    discount_case, discount_label, derivations = _resolve_discount(df, mappings)

    # ---- capability matrix ---------------------------------------------------
    capabilities = _capability_matrix(mappings, derivations)

    # ---- agent reasoning -----------------------------------------------------
    agent_reasoning = _build_reasoning(mappings, discount_case, capabilities)

    all_concepts = list(SEMANTIC_HINTS.keys())
    missing_concepts = [c for c in all_concepts if c not in mappings]

    return {
        "mappings": mappings,
        "derivations": derivations,
        "discount_case": discount_case,
        "discount_label": discount_label,
        "capabilities": capabilities,
        "agent_reasoning": agent_reasoning,
        "missing_concepts": missing_concepts,
        "columns": columns,
        "dtypes": dtypes,
        "row_count": int(len(df)),
    }


def _safe_sample(series: pd.Series) -> Any:
    s = series.dropna()
    if len(s) == 0:
        return None
    v = s.iloc[0]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 4)
    return str(v)[:60]


def _resolve_discount(df: pd.DataFrame, mappings: Dict[str, Dict]) -> (Optional[int], Optional[str], List[Dict]):
    """Determine which of the 8 discount cases applies and derive canonical fields."""
    derivations: List[Dict] = []
    has = lambda c: c in mappings  # noqa: E731

    def col_of(concept):
        return mappings[concept]["column"] if has(concept) else None

    # CASE 1 — original + final price
    if has("original_value") and has("final_value"):
        oc, fc = col_of("original_value"), col_of("final_value")
        derivations.append({"field": "discount_amount", "from": [oc, fc],
                            "formula": f"{oc} - {fc}"})
        derivations.append({"field": "discount_percent", "from": [oc, fc],
                            "formula": f"({oc} - {fc}) / {oc}"})
        derivations.append({"field": "treated", "from": [oc, fc],
                            "formula": f"({oc} - {fc}) > 0"})
        return 1, "Original + Final price", derivations

    # CASE 2 — original + discount percent
    if has("original_value") and has("discount_percent"):
        oc, pc = col_of("original_value"), col_of("discount_percent")
        derivations.append({"field": "discount_amount", "from": [oc, pc],
                            "formula": f"{oc} * {pc}"})
        derivations.append({"field": "final_value", "from": [oc, pc],
                            "formula": f"{oc} * (1 - {pc})"})
        derivations.append({"field": "treated", "from": [pc],
                            "formula": f"{pc} > 0"})
        return 2, "Original price + Discount percentage", derivations

    # CASE 3 — original + discount amount
    if has("original_value") and has("discount_amount"):
        oc, ac = col_of("original_value"), col_of("discount_amount")
        derivations.append({"field": "discount_percent", "from": [oc, ac],
                            "formula": f"{ac} / {oc}"})
        derivations.append({"field": "final_value", "from": [oc, ac],
                            "formula": f"{oc} - {ac}"})
        derivations.append({"field": "treated", "from": [ac],
                            "formula": f"{ac} > 0"})
        return 3, "Original price + Discount amount", derivations

    # CASE 4 — discount amount only
    if has("discount_amount"):
        ac = col_of("discount_amount")
        derivations.append({"field": "treated", "from": [ac],
                            "formula": f"{ac} > 0"})
        return 4, "Discount amount only", derivations

    # CASE 5 — discount percent only
    if has("discount_percent"):
        pc = col_of("discount_percent")
        derivations.append({"field": "treated", "from": [pc],
                            "formula": f"{pc} > 0"})
        return 5, "Discount percentage only", derivations

    # CASE 6 — treatment indicator
    if has("treatment"):
        tc = col_of("treatment")
        derivations.append({"field": "treated", "from": [tc],
                            "formula": f"bool({tc}) truthy"})
        return 6, "Coupon / treatment indicator", derivations

    # CASE 7 — coupon code
    if has("coupon_code"):
        cc = col_of("coupon_code")
        derivations.append({"field": "treated", "from": [cc],
                            "formula": f"{cc} not in (none/empty)"})
        return 7, "Coupon code", derivations

    # CASE 8 — no discount info
    return 8, "No discount information", derivations


def _capability_matrix(mappings: Dict, derivations: List[Dict]) -> Dict[str, Dict[str, str]]:
    has = lambda c: c in mappings  # noqa: E731
    derived_fields = {d["field"] for d in derivations}
    can = lambda c: has(c) or c in derived_fields  # noqa: E731

    treatment_present = can("treated")
    money_present = can("discount_amount") or can("final_value") or can("original_value")
    exact_spend = can("discount_amount") or (has("original_value") and has("final_value")) \
        or (has("original_value") and has("discount_percent"))
    profit_present = has("profit")
    conversion_present = has("conversion")
    behavioral = any(has(b) for b in
                     ("dwell_time", "product_views", "price_comparison", "cart_added",
                      "days_since_last_visit"))

    def status(ok, reason_ok, reason_no):
        return "READY" if ok else ("LIMITED" if False else "NEEDS_DATA"), reason_ok if ok else reason_no

    caps: Dict[str, Dict[str, str]] = {}

    caps["discount_impact"] = _cap(treatment_present and conversion_present,
        "Treatment + conversion detected — discount behavior can be analyzed.",
        "Needs a treatment/discount signal and a conversion outcome.")

    caps["exact_discount_spend"] = _cap(exact_spend,
        "Monetary discount fields present — exact spend calculable.",
        "Needs a discount amount, or original+final price, or original+discount%.")

    caps["customer_segmentation"] = _cap(conversion_present or behavioral or treatment_present,
        "Conversion/behavioral signals detected — segmentation possible.",
        "Needs conversion, behavioral features, or a treatment signal.")

    caps["behavioral_opportunity"] = _cap(behavioral or conversion_present,
        "Behavioral or conversion signals available for opportunity estimate.",
        "Needs behavioral features or a conversion outcome.")

    caps["profit_recovery"] = _cap(profit_present,
        "Profit/margin column detected — recovery estimate is margin-based.",
        "Needs a profit/margin column for a margin-based recovery estimate.")

    caps["margin_leakage"] = _cap(profit_present and treatment_present and conversion_present,
        "Profit + treatment + conversion present — leakage can be attributed.",
        "Needs profit + a treatment signal + conversion.")

    return caps


def _cap(ok: bool, reason_ok: str, reason_no: str) -> Dict[str, str]:
    return {"status": "READY" if ok else "NEEDS_DATA",
            "reason": reason_ok if ok else reason_no}


def _build_reasoning(mappings: Dict, discount_case: Optional[int],
                     capabilities: Dict) -> List[str]:
    has = lambda c: c in mappings  # noqa: E731
    steps: List[str] = []

    found = []
    if has("customer_id"):
        found.append(f"customer identity (`{mappings['customer_id']['column']}`)")
    if has("conversion"):
        found.append(f"purchase outcome (`{mappings['conversion']['column']}`)")
    if has("treatment"):
        found.append(f"discount exposure (`{mappings['treatment']['column']}`)")
    if has("profit"):
        found.append(f"profit/margin (`{mappings['profit']['column']}`)")
    if found:
        steps.append("I found " + ", ".join(found) + ".")
    else:
        steps.append("I inspected the columns but could not match key business concepts.")

    if discount_case in (1, 2, 3):
        steps.append("Discount value is recoverable from the price fields, "
                     "so exact discount spend can be calculated.")
    elif discount_case in (4, 5):
        steps.append("A discount amount/percentage is present, but without an original "
                     "price the exact monetary spend is only partially known.")
    elif discount_case == 6:
        steps.append("Discount exposure is detected via a treatment indicator, but the "
                     "exact discount value is unavailable. I can still analyze customer "
                     "response and treatment patterns.")
    elif discount_case == 7:
        steps.append("A coupon-code column was found. I can infer who received a discount, "
                     "but exact discount spend cannot be calculated without coupon values.")
    elif discount_case == 8:
        steps.append("I could not identify a discount or treatment variable in this dataset. "
                     "Discount-impact analysis is not possible without one of: a treatment "
                     "indicator, coupon usage, discount amount, discount percentage, or "
                     "original+final price.")

    ready = [k for k, v in capabilities.items() if v["status"] == "READY"]
    if ready:
        steps.append("Based on what's present, your data can support: " + ", ".join(ready) + ".")
    return steps


# ---- resolution: build a canonical dataframe for the analyzer ----------------
def resolve_dataframe(df: pd.DataFrame, schema: Dict[str, Any]) -> pd.DataFrame:
    """
    Produce a canonical dataframe the analyzer can consume, deriving discount fields
    ONLY from defensible derivations. Columns that cannot be derived are simply absent.
    """
    out = df.copy()
    mappings = schema["mappings"]
    derivations = schema["derivations"]
    has = lambda c: c in mappings  # noqa: E731

    def col_of(concept):
        return mappings[concept]["column"] if has(concept) else None

    # rename mapped columns to canonical concepts
    rename = {v["column"]: k for k, v in mappings.items()}
    out = out.rename(columns=rename)

    def to_num(series):
        return pd.to_numeric(series, errors="coerce")

    # apply derivations in dependency order
    case = schema.get("discount_case")
    if case == 1 and has("original_value") and has("final_value"):
        out["discount_amount"] = to_num(out["original_value"]) - to_num(out["final_value"])
        out["discount_percent"] = (out["discount_amount"] / to_num(out["original_value"])).clip(lower=0)
        out["treated"] = out["discount_amount"] > 0
    elif case == 2 and has("original_value") and has("discount_percent"):
        out["discount_percent"] = to_num(out["discount_percent"]).clip(lower=0)
        out["discount_amount"] = to_num(out["original_value"]) * out["discount_percent"]
        out["final_value"] = to_num(out["original_value"]) - out["discount_amount"]
        out["treated"] = out["discount_percent"] > 0
    elif case == 3 and has("original_value") and has("discount_amount"):
        out["discount_amount"] = to_num(out["discount_amount"]).clip(lower=0)
        out["discount_percent"] = out["discount_amount"] / to_num(out["original_value"])
        out["final_value"] = to_num(out["original_value"]) - out["discount_amount"]
        out["treated"] = out["discount_amount"] > 0
    elif case == 4 and has("discount_amount"):
        out["discount_amount"] = to_num(out["discount_amount"]).clip(lower=0)
        out["treated"] = out["discount_amount"] > 0
    elif case == 5 and has("discount_percent"):
        out["discount_percent"] = to_num(out["discount_percent"]).clip(lower=0)
        out["treated"] = out["discount_percent"] > 0
    elif case == 6 and has("treatment"):
        out["treated"] = _truthy(out["treatment"])
    elif case == 7 and has("coupon_code"):
        out["treated"] = ~out["coupon_code"].astype(str).str.strip().str.lower().isin(["none", "nan", "", "no", "null"])

    # normalize discount_percent to a 0..1 fraction if it looks like a percentage
    if "discount_percent" in out.columns:
        dp = to_num(out["discount_percent"])
        if dp.max() > 1.5:
            out["discount_percent"] = dp / 100.0
        else:
            out["discount_percent"] = dp

    return out


def _truthy(series: pd.Series) -> pd.Series:
    """Coerce a treatment-indicator column to boolean without inventing values."""
    s = series.copy()
    # numeric: >0 is treated
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.7:
        return num > 0
    # otherwise string truthiness
    low = s.astype(str).str.strip().str.lower()
    return ~low.isin(["0", "false", "no", "none", "nan", "", "n", "f"])
