"""Discount Lens — end-to-end system tests.

Covers: demo load, arbitrary CSV, column mapping, invalid CSV, missing
conversion rejection, model training, predictions, segmentation, leakage,
simulator sensitivity, optimizer, deployment, agent state, reload
consistency, and metric reconciliation.
"""

import io
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.simplefilter("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.demo_data import generate_demo_dataset
from agent.agent_analyzer import CausalAnalyzer
from agent.agent_recommender import RevenueRecommender
from agent.agent_simulator import simulate_policy, DEFAULT_POLICY
from agent.agent_memory import AgentMemory
from agent.agent_orchestrator import RevenueAgent
from agent.agent_tools import policy_json

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


def run():
    print("=== Discount Lens System Tests ===\n")

    # 1. Demo dataset loads
    print("[1] Demo dataset loads")
    df = generate_demo_dataset(rows=500, seed=11)
    check("Rows generated", len(df) == 500)
    check("Has treated", "treated" in df.columns)
    check("Has conversion", "conversion" in df.columns)
    check("Has behavioral features", "product_views" in df.columns and "dwell_time_seconds" in df.columns)
    check("Has profit", "profit" in df.columns)
    check("Has dimensions", all(c in df.columns for c in ("channel", "campaign_id", "product_category", "region")))

    # 2. Arbitrary CSV loads (with aliases)
    print("\n[2] Arbitrary CSV with aliases loads")
    alias_df = df.rename(columns={
        "customer_id": "user_id", "treated": "coupon_used", "conversion": "purchased",
        "original_value": "order_value", "discount_amount": "discount_value", "profit": "margin",
    })
    # simulate schema detection by renaming to canonical
    from schema_engine import detect_schema, resolve_dataframe
    schema = detect_schema(alias_df)
    check("Schema detects aliases", len(schema.get("mappings", {})) > 0, f"mappings={schema.get('mappings')}")

    # 3. Column mapping works (manual rename)
    print("\n[3] Column mapping works")
    manual = alias_df.rename(columns={
        "user_id": "customer_id", "coupon_used": "treated", "purchased": "conversion",
        "order_value": "original_value", "discount_value": "discount_amount", "margin": "profit",
    })
    v = CausalAnalyzer().validate(manual)
    check("Mapped CSV is causal-ready", v.ready, detail="; ".join(f["message"] for f in v.findings if f["level"] == "error"))

    # 4. Invalid CSV produces useful error
    print("\n[4] Invalid CSV produces useful error")
    bad_df = pd.DataFrame({"junk": [1, 2], "nothing": ["a", "b"]})
    v_bad = CausalAnalyzer().validate(bad_df)
    check("Not ready", not v_bad.ready)
    check("Error message present", any("needs" in f["message"].lower() or "required" in f["message"].lower() or "treatment" in f["message"].lower() for f in v_bad.findings))

    # 5. Missing conversion produces useful error
    print("\n[5] Missing conversion produces useful error")
    no_conv = df.drop(columns=["conversion"])
    v_nc = CausalAnalyzer().validate(no_conv)
    check("Not ready without conversion", not v_nc.ready)
    check("Mentions conversion", any("conversion" in f["message"].lower() for f in v_nc.findings))

    # 6. Model trains
    print("\n[6] Model trains")
    a = CausalAnalyzer()
    v = a.validate(df)
    model = a.analyze(df, v)
    check("Model available", model.get("available", False), detail=model.get("message", ""))
    check("Has diagnostics", bool(model.get("diagnostics")))

    # 7. Predictions generated
    print("\n[7] Predictions generated")
    preds = model.get("predictions", [])
    check("Predictions non-empty", len(preds) == len(df))
    check("Has baseline prob", "baseline_probability" in preds[0] if preds else False)
    check("Has treated prob", "treated_probability" in preds[0] if preds else False)
    check("Has incremental lift", "estimated_incremental_lift" in preds[0] if preds else False)

    # 8. Segments generated
    print("\n[8] Segments generated")
    segs = set(r.get("customer_type") for r in preds)
    check("Sure Thing present", "Sure Thing" in segs)
    check("Persuadable present", "Persuadable" in segs)
    check("Lost Cause or Price Sensitive present", bool(segs & {"Lost Cause", "Price Sensitive"}))

    # 9. Leakage calculated
    print("\n[9] Leakage calculated")
    leak_rows = [r for r in preds if r.get("was_leakage")]
    check("Leakage rows exist", len(leak_rows) > 0)
    total_leak = sum(r.get("discount_amount_canonical", 0) for r in leak_rows)
    check("Leakage amount > 0", total_leak > 0, detail=f"leakage={total_leak}")

    # 10. Simulator changes when policy changes
    print("\n[10] Simulator changes when policy changes")
    state = {"model": model}
    sim_current = simulate_policy(DEFAULT_POLICY, DEFAULT_POLICY, state)
    sim_proposed = simulate_policy(DEFAULT_POLICY, {"sure_thing_discount_cap": 0.0, "persuadable_max": 0.10, "price_sensitive_max": 0.10}, state)
    check("Current != proposed profit", sim_current["current"]["profit"] != sim_proposed["proposed"]["profit"])
    check("Delta present", "profit_delta" in sim_proposed["deltas"])

    # 11. Optimizer finds a policy
    print("\n[11] Optimizer finds a policy")
    rec = RevenueRecommender()
    opt = rec.optimize(simulate_policy, state)
    check("No error", "error" not in opt, detail=opt.get("error", ""))
    check("Policy returned", bool(opt.get("policy")))
    check("Simulation present", bool(opt.get("simulation")))
    check("Revenue constraint met", opt["simulation"]["proposed"]["revenue"] >= opt["simulation"]["current"]["revenue"] * 0.94)

    # 12. Policy deployment works
    print("\n[12] Policy deployment works")
    mem = AgentMemory()
    mem.start("test", state)
    deployed = mem.deploy("test", opt["policy"])
    check("Deployed has timestamp", "deployed_at" in deployed)
    check("Deployed policy stored", mem.get("test").get("deployed_policy") is not None)
    pj = policy_json(opt["policy"])
    check("Policy JSON has rules", "rules" in pj and len(pj["rules"]) >= 2)
    check("Policy JSON version", pj["policy_version"] == "1.0")

    # 13. Agent answers based on actual state
    print("\n[13] Agent answers based on actual state")
    agent = RevenueAgent(mem)
    full_state = agent.build_state("test", {"row_count": len(df), "columns": list(df.columns), "mappings": v.feature_columns}, v.as_dict(), model, {})
    ans = agent.investigate("test", "which channel leaks the most?", full_state)
    check("Answer has finding", bool(ans.get("finding")))
    check("Answer has evidence", len(ans.get("evidence", [])) > 0)
    check("Uses channel tool", "analyze_channel" in ans.get("tools_used", []))

    # 13b. Memory follow-up
    print("\n[13b] Agent memory follow-up")
    ans2 = agent.investigate("test", "what should we do about it?", full_state)
    check("Follow-up has recommendation", bool(ans2.get("recommendation")))
    check("Focus retained", mem.get("test").get("focus", {}).get("dimension") == "channel")

    # 14. Reloading doesn't create inconsistent numbers
    print("\n[14] Reload consistency")
    model2 = CausalAnalyzer().analyze(df, CausalAnalyzer().validate(df))
    check("Same prediction count", len(model2["predictions"]) == len(model["predictions"]))
    check("Same reliability", model2["reliability"] == model["reliability"])

    # 15. Metric reconciliation
    print("\n[15] Metric reconciliation")
    from agent.agent_tools import rows_from_predictions, dimension_breakdown
    rows = rows_from_predictions(model)
    total_leak_df = float(rows.loc[rows["was_leakage"], "discount_amount_canonical"].sum())
    total_leak_pred = sum(r.get("discount_amount_canonical", 0) for r in preds if r.get("was_leakage"))
    check("Leakage reconciles (predictions vs frame)", abs(total_leak_df - total_leak_pred) < 0.01, detail=f"df={total_leak_df}, pred={total_leak_pred}")
    channel_leak = sum(d["leakage"] for d in dimension_breakdown(rows, "channel"))
    check("Channel leakage sums to total", abs(channel_leak - total_leak_df) < 0.01, detail=f"channel_sum={channel_leak}, total={total_leak_df}")

    print(f"\n=== Results: {PASS} passed, {FAIL} failed ===")
    return FAIL == 0


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
