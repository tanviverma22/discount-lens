# Discount Lens

**"See the discounts that were never needed."**

Discount Lens is an agentic commerce product that finds discount leakage — the margin wasted on customers who would have purchased anyway — and turns it into recoverable profit.

Instead of the classic "upload CSV → dashboard appears" flow, Discount Lens feels like handing your data to an AI analyst. The agent inspects the schema, validates data quality, fits a causal uplift model, segments customers, ranks recovery opportunities, and recommends — then simulates and deploys — a new discount policy.

---

## The problem

E-commerce teams measure promotions by revenue, orders, conversion rate, and campaign ROI. None of those answer the only question that matters:

> **Did the discount actually cause the customer to buy?**

Some customers need the incentive. Others would have bought anyway — their discount just erodes margin without changing the outcome. Discount Lens separates those two groups and quantifies the difference.

The goal is not "give fewer discounts." The goal is **"give discounts where they create incremental value."**

---

## What it does

| Capability | Description |
|---|---|
| **Proactive agent** | Produces a Daily Revenue Intelligence Brief with ranked recovery opportunities — no prompting required |
| **Investigation trail** | A visible OBSERVE → VALIDATE → MODEL → CHECK → INVESTIGATE → RECOMMEND workflow |
| **Causal uplift model** | T-learner (propensity + treatment/control outcome models) with common-support diagnostics and a reliability grade |
| **Uplift segmentation** | Sure Things, Persuadables, Price Sensitive, Lost Causes — derived from estimated incremental profit, not hardcoded rules |
| **Tool-driven chat** | Plain-English questions routed to data tools, with session memory |
| **Policy simulation** | Side-by-side financial impact of current vs. proposed discount policy |
| **Policy optimizer** | Searches policy space for max profit under a 95% revenue-retention constraint |
| **Policy deployment** | Versioned policy JSON saved as the active policy (demo-local) |
| **Demo data generator** | Reproducible 2,500-row causal dataset — no hardcoded dashboard values |

---

## Quick start

### 1. Create a virtual environment

```bash
cd discountlens
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

The app starts on **http://127.0.0.1:5001/**.

### 4. Load the demo and explore

1. Open the app and click the **Revenue Agent** tab.
2. Click **Load Demo Dataset** — the agent runs its investigation and produces the Daily Brief.
3. Ask it questions in the **Agent Workspace** (e.g. `which channel leaks the most?`).
4. Click **Find Best Policy** to optimize the discount policy, then **Deploy**.

---

## Project structure

```
discountlens/
├── app.py                 # Flask app, routes, session management
├── ai_agent.py            # Keyword-routed "investigator" (legacy)
├── discount_analyzer.py   # Core discount-leakage analysis
├── schema_engine.py       # Schema detection, column mapping, data health
├── agent/                 # Revenue Agent package
│   ├── agent_analyzer.py      # CausalAnalyzer: validation + T-learner uplift
│   ├── agent_orchestrator.py  # RevenueAgent: trail, brief, investigate
│   ├── agent_recommender.py   # Opportunity ranking + policy optimization
│   ├── agent_simulator.py     # Policy financial simulation
│   ├── agent_tools.py         # Data tools: breakdowns, evidence, policy JSON
│   ├── agent_memory.py        # Per-session conversation memory
│   └── demo_data.py           # Causal-ready demo dataset generator
├── data/
│   ├── sample_transactions.csv
│   └── causal_demo_dataset.csv
├── templates/
│   └── index.html         # Single-page dashboard UI
├── docs/
│   └── business-model.md  # Business model & go-to-market
├── test_system.py         # End-to-end system tests
└── requirements.txt
```

---

## How the causal engine works

1. **Validate** — requires `customer_id`, `treated`, `conversion`; checks treatment/control balance, minimum group sizes, and feature availability.
2. **Preprocess** — encodes categoricals, imputes missing values, scales numerics.
3. **Fit** — a propensity model (logistic regression on features) plus separate treated and control outcome models.
4. **Estimate uplift** — `lift = P(convert | treated) − P(convert | control)` per observation.
5. **Diagnose** — training/holdout split, treated/control counts, common-support share, and a HIGH/MODERATE/LOW reliability grade.

Expected incremental profit is `lift × margin − discount_cost`, which drives segmentation and opportunity ranking.

---

## API surface

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/agent/demo` | POST | Generate + run analysis on the demo dataset |
| `/api/agent/state` | GET | Full agent state for the session |
| `/api/agent/brief` | GET | Proactive Daily Revenue Intelligence Brief |
| `/api/agent/investigate-v2` | POST | Ask the agent a question |
| `/api/agent/simulate` | POST | Simulate a proposed discount policy |
| `/api/agent/optimize` | POST | Search for the best policy |
| `/api/agent/deploy` | POST | Save a policy as active |
| `/api/system-check` | GET | Self-check all components |
| `/api/analyze` | POST | Run discount-leakage analysis |
| `/api/upload` | POST | Upload a CSV |
| `/dataset` | GET | Readable HTML preview of the loaded dataset |

---

## Business model

See [`docs/business-model.md`](docs/business-model.md) for the full business model, unit economics, go-to-market, and roadmap.

---

## License

Proprietary — this is a hackathon demo. Contact the author before reuse.
