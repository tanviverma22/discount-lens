# Discount Lens

> **Agentic discount-leakage detection for e-commerce.**
> See the discounts that were never needed — and recover the margin they waste.

---

## Table of contents

1. [What problem are we solving?](#what-problem-are-we-solving)
2. [Why existing dashboards fail](#why-existing-dashboards-fail)
3. [The Discount Lens solution](#the-discount-lens-solution)
4. [How it works](#how-it-works)
5. [Key capabilities](#key-capabilities)
6. [Business impact](#business-impact)
7. [Installation & quick start](#installation--quick-start)
8. [Using the product](#using-the-product)
9. [Architecture](#architecture)
10. [API surface](#api-surface)
11. [Data model](#data-model)
12. [How the causal engine works](#how-the-causal-engine-works)
13. [Project structure](#project-structure)
14. [Business model](#business-model)
15. [Roadmap](#roadmap)
16. [License](#license)

---

## What problem are we solving?

E-commerce teams run promotions constantly: welcome codes, flash sales, cart-abandonment coupons, loyalty rewards, affiliate discounts. The dashboards look good — revenue is up, orders are up, conversion rate is up.

But the real question is almost never answered:

> **Did the discount actually cause the customer to buy, or did it just shrink the margin on a sale that was going to happen anyway?**

That second case is called **discount leakage**. It happens when customers who are already motivated to purchase — "sure things" — still receive a discount. Every leaked discount is pure margin destruction with zero incremental revenue.

Discount leakage is expensive and invisible:
- It does not show up as a campaign failure.
- It inflates short-term revenue while eroding unit economics.
- It trains customers to wait for coupons before buying.
- It hides inside aggregated conversion numbers.

**Discount Lens finds the leakage and turns it into a recoverable profit opportunity.**

---

## Why existing dashboards fail

Traditional analytics optimize for metrics that reward giving away margin:

| Metric | Why it hides leakage |
|---|---|
| **Conversion rate** | Tells you *that* a segment bought, not *because of* the discount. |
| **Revenue / GMV** | Includes full-price buyers who were mislabeled as discount-attributed. |
| **Campaign ROI** | Compares treated customers to a vague baseline, not a matched counterfactual. |
| **A/B test uplift** | Correct in principle, but most teams cannot A/B test every discount permutation at scale. |
| **Cohort LTV** | Too slow; leakage decisions need daily operational feedback. |

The missing capability is **causal inference at the transaction level**: estimating, for each individual customer, what they would have done *without* the discount.

Discount Lens does exactly that.

---

## The Discount Lens solution

Discount Lens is an **agentic revenue analyst** that:

1. **Observes** the uploaded transaction data and learns its schema.
2. **Validates** data quality, treatment/control balance, and causal feasibility.
3. **Models** the incremental effect of each discount (causal uplift) on every customer.
4. **Segments** customers into actionable groups: Sure Things, Persuadables, Price Sensitive, Lost Causes.
5. **Ranks** recovery opportunities by recoverable margin.
6. **Recommends** a tighter discount policy.
7. **Simulates** the financial impact before deployment.
8. **Deploys** the new policy (demo-local) as a versioned JSON artifact.

Instead of a static dashboard, the user talks to a Revenue Agent that explains its reasoning, exposes its methodology, and can answer follow-up questions about the data.

---

## How it works

### High-level flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  USER: uploads CSV or clicks "Load Demo Dataset"                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  OBSERVE    Schema engine detects columns, maps synonyms, flags missing       │
│             required fields.                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  VALIDATE   CausalAnalyzer checks treatment/control balance, minimum sample   │
│             size, feature availability, and issues a reliability grade.       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL      T-learner fits:                                                   │
│             • propensity score P(treated | features)                          │
│             • treated outcome P(conversion | treated, features)               │
│             • control outcome   P(conversion | control, features)             │
│             Uplift = treated probability − control probability                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  CHECK      Common-support filter, holdout diagnostics, reliability summary.  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  INVESTIGATE Agent workspace answers plain-English questions by routing to    │
│             data tools: channel breakdowns, customer counts, policy impact,   │
│             leakage evidence, etc.                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  RECOMMEND  Recommender ranks Sure-Thing leakage and proposes an optimized    │
│             discount policy under a revenue-retention constraint.             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIMULATE   User previews current vs. proposed P&L before committing.         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DEPLOY     Versioned policy JSON is saved as the active discount policy.     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### What the user sees

- A single-page web app with tabs for Dashboard, Agent Workspace, Policy Simulation, and Dataset Preview.
- A **Daily Revenue Intelligence Brief** generated automatically when demo data loads.
- A chat interface where natural-language questions trigger data tools.
- Visual outputs: leakage by channel/segment, uplift distribution, policy comparison.

---

## Key capabilities

| Capability | What it gives the business |
|---|---|
| **Proactive Daily Brief** | Wakes up to ranked recovery opportunities without writing SQL. |
| **Investigation trail** | Visible OBSERVE → VALIDATE → MODEL → CHECK → INVESTIGATE → RECOMMEND pipeline. |
| **Causal uplift modeling** | T-learner with propensity + outcome models, common-support diagnostics, reliability grade. |
| **Uplift segmentation** | Sure Things, Persuadables, Price Sensitive, Lost Causes — derived from predicted incremental profit. |
| **Tool-driven chat** | Ask "which channel leaks the most?" and get a real query result, not a hallucination. |
| **Policy optimizer** | Searches discount space for max profit under a 95% revenue-retention guardrail. |
| **Policy simulator** | Side-by-side current vs. proposed P&L before deployment. |
| **Policy deployment** | Versioned active policy JSON (demo-local storage). |
| **Demo data generator** | Reproducible 2,500-row causal dataset so every judge sees the same end-to-end demo. |
| **Self-check endpoint** | `/api/system-check` verifies the whole stack on demand. |

---

## Business impact

### The leakage equation

For each customer:

```
Incremental profit = uplift × margin − discount_cost
```

Where:
- `uplift` = extra probability of buying because of the discount.
- `margin` = average gross margin per order.
- `discount_cost` = discount value given to the customer.

If `uplift` is near zero and `discount_cost` is positive, the discount is **leakage**.

### Example numbers from a typical demo run

| Segment | Definition | Action | Recoverable value |
|---|---|---|---|
| **Sure Things** | Would buy anyway; received discount | Stop/discount less | High — pure margin recovery |
| **Persuadables** | Bought because of discount | Keep discount | High — protect incremental revenue |
| **Price Sensitive** | Need discount to convert | Consider smaller discount | Medium — optimize cost |
| **Lost Causes** | Will not convert either way | Stop discounting | Low — save spend |

A 10% reduction in Sure-Thing discounts often translates to **2–5 points of operating margin** on promotional revenue — without losing sales.

---

## Installation & quick start

### 1. Clone / enter the project

```bash
cd discountlens
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Core dependencies: Flask, pandas, scikit-learn, numpy, flask-cors.

### 4. Run the app

```bash
python app.py
```

The server starts on **http://127.0.0.1:5001/**.

### 5. Health check

```bash
curl http://127.0.0.1:5001/api/health
curl http://127.0.0.1:5001/api/system-check
```

---

## Using the product

### First-time demo (recommended)

1. Open **http://127.0.0.1:5001/**.
2. Click the **Revenue Agent** tab.
3. Click **Load Demo Dataset**.
4. Read the **Daily Revenue Intelligence Brief**.
5. Ask questions in the chat, for example:
   - `which channel leaks the most?`
   - `how many sure things got discounts?`
   - `what is the total recoverable margin?`
   - `show me the policy comparison`
6. Go to **Policy Simulation**, click **Find Best Policy**, then **Deploy**.
7. Watch the active policy and the projected P&L update.

### Uploading your own data

The CSV must contain:

| Required column | Meaning |
|---|---|
| `customer_id` | Unique customer identifier |
| `treated` | 1 if customer received a discount, 0 otherwise |
| `conversion` | 1 if customer purchased, 0 otherwise |

Recommended columns for richer analysis:

| Column | Purpose |
|---|---|
| `channel` | Acquisition channel (email, organic, paid, affiliate, etc.) |
| `segment` | Pre-defined business segment |
| `discount_value` | Dollar value of discount given |
| `order_value` | Transaction value |
| `margin` | Gross margin for the transaction |
| `aov`, `ltv`, `tenure`, `rfm_score` | Customer value signals |
| `product_category`, `region`, `device` | Context features |

If your columns have different names, the schema engine attempts synonym mapping automatically.

---

## Architecture

```
 discountlens/
 ├── app.py                 Flask application, routes, session state
 ├── ai_agent.py            Legacy keyword-routed investigator
 ├── discount_analyzer.py   Core discount-leakage calculations
 ├── schema_engine.py       Schema detection, mapping, data-health checks
 ├── agent/                 New Revenue Agent package
 │   ├── agent_analyzer.py      CausalAnalyzer: validation + T-learner
 │   ├── agent_orchestrator.py  RevenueAgent: trail, brief, investigation
 │   ├── agent_recommender.py   Opportunity ranking + policy optimization
 │   ├── agent_simulator.py     Financial policy simulation
 │   ├── agent_tools.py         Data tools the agent can call
 │   ├── agent_memory.py        Per-session conversation memory
 │   └── demo_data.py           Reproducible causal dataset generator
 ├── templates/
 │   └── index.html         Single-page dashboard UI
 ├── data/
 │   ├── sample_transactions.csv
 │   └── causal_demo_dataset.csv
 ├── docs/
 │   └── business-model.md  Business model, unit economics, go-to-market
 ├── test_system.py         End-to-end system tests
 └── requirements.txt
```

---

## API surface

| Route | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness check |
| `/api/system-check` | GET | Verifies data, schema, model, and agent components |
| `/api/upload` | POST | Upload a CSV file |
| `/api/analyze` | POST | Run the classic discount-leakage analysis |
| `/api/agent/demo` | POST | Generate + analyze the demo dataset |
| `/api/agent/state` | GET | Full agent state for the current session |
| `/api/agent/brief` | GET | Daily Revenue Intelligence Brief |
| `/api/agent/investigate-v2` | POST | Ask the agent a question |
| `/api/agent/simulate` | POST | Simulate a proposed discount policy |
| `/api/agent/optimize` | POST | Search for the best policy |
| `/api/agent/deploy` | POST | Save a policy as the active policy |
| `/dataset` | GET | Human-readable HTML preview of the loaded dataset |

### Example: ask the agent a question

```bash
curl -X POST http://127.0.0.1:5001/api/agent/investigate-v2 \
  -H "Content-Type: application/json" \
  -d '{"question": "which channel leaks the most?"}'
```

### Example: optimize policy

```bash
curl -X POST http://127.0.0.1:5001/api/agent/optimize
```

---

## Data model

The demo dataset (`data/causal_demo_dataset.csv`) contains 2,500 rows with the following structure:

| Column | Type | Description |
|---|---|---|
| `customer_id` | int | Unique customer identifier |
| `treated` | int {0,1} | Whether the customer received a discount |
| `conversion` | int {0,1} | Whether the customer purchased |
| `channel` | string | Acquisition channel |
| `segment` | string | Customer segment |
| `device` | string | Device type |
| `region` | string | Geographic region |
| `tenure_months` | int | How long the customer has been active |
| `past_orders` | int | Historical order count |
| `avg_order_value` | float | Historical AOV |
| `discount_value` | float | Discount amount offered |
| `order_value` | float | Order value if converted |
| `margin` | float | Gross margin if converted |

The `treated` and `conversion` flags are constructed to create realistic heterogeneity: some customers are truly persuadable, others are sure things, and the rest are lost causes. This lets the causal model demonstrate actionable uplift segmentation.

---

## How the causal engine works

Discount Lens uses a **T-learner** causal uplift model, the standard approach when treatment effects are heterogeneous.

### 1. Validation

Before modeling, `CausalAnalyzer.validate(df)` checks:
- Required columns exist (`customer_id`, `treated`, `conversion`).
- Treatment and control groups are large enough (≥ 50 each).
- Treatment fraction is between 10% and 90%.
- There is at least one meaningful feature beyond the required columns.
- Baseline conversion rate is not pathological.

If validation passes, it emits a reliability grade (`HIGH`, `MODERATE`, or `LOW`) based on balance, sample size, and common support.

### 2. Preprocessing

- Drop ID columns and non-feature fields.
- One-hot encode categoricals.
- Impute missing values with column medians.
- Scale numeric features.

### 3. Model fitting

Three models are trained:

| Model | Target | Interpretation |
|---|---|---|
| Propensity model | `P(treated \| X)` | How likely each customer was to receive the discount |
| Treated outcome model | `P(conversion \| treated, X)` | Conversion probability under discount |
| Control outcome model | `P(conversion \| control, X)` | Conversion probability without discount |

All use scikit-learn `LogisticRegression` with regularization for stability.

### 4. Uplift estimation

For each customer:

```python
uplift = P(conversion | treated, X) - P(conversion | control, X)
```

Uplift is bounded to the valid probability interval `[-1, 1]`.

### 5. Incremental profit

```python
incremental_profit = uplift * margin - discount_cost
```

This is the key business score. Customers with high incremental profit are Persuadables; customers with negative incremental profit are Sure Things leaking margin.

### 6. Segmentation

Based on uplift and incremental profit:

| Segment | Condition | Business action |
|---|---|---|
| **Sure Things** | High control probability, low uplift | Remove or reduce discount |
| **Persuadables** | High uplift, positive incremental profit | Maintain/keep discount |
| **Price Sensitive** | Moderate uplift, low margin | Test smaller discount |
| **Lost Causes** | Low uplift, low control probability | Stop discounting |

### 7. Diagnostics

Every analysis returns:
- Treated count, control count.
- Training/holdout split.
- Common-support share (customers whose propensity score is not extreme).
- Reliability grade.

---

## Project structure

```
discountlens/
├── app.py                      Flask app and REST API
├── ai_agent.py                 Legacy keyword-routed agent wrapper
├── discount_analyzer.py        Discount leakage calculations
├── schema_engine.py            Schema inference and data health
├── agent/                      New agentic layer
│   ├── __init__.py
│   ├── agent_analyzer.py       Causal validation and T-learner uplift
│   ├── agent_orchestrator.py   RevenueAgent workflow and brief
│   ├── agent_recommender.py    Opportunity ranking + policy optimizer
│   ├── agent_simulator.py      Policy simulation engine
│   ├── agent_tools.py          Data tools invoked by the agent
│   ├── agent_memory.py         Session memory
│   └── demo_data.py            Reproducible causal demo data
├── data/
│   ├── sample_transactions.csv
│   └── causal_demo_dataset.csv
├── docs/
│   └── business-model.md       Business model and go-to-market
├── templates/
│   └── index.html              Single-page dashboard
├── static/                     Static assets
├── test_system.py              End-to-end system tests
├── requirements.txt
└── README.md
```

---

## Business model

See [`docs/business-model.md`](docs/business-model.md) for the full business model including:
- Target customer profiles.
- Pricing (SaaS tiering, implementation, managed services).
- Unit economics and ROI math.
- Go-to-market motion.
- Roadmap.

---

## Roadmap

| Phase | Feature | Goal |
|---|---|---|
| **Now** | End-to-end demo with causal agent + policy optimizer | Prove the concept and win hackathon/demo interest |
| **Next** | Real-time inference API, model retraining pipeline | Move from batch analysis to production scoring |
| **Soon** | Integration with Shopify / BigCommerce / Stripe | One-click data connection for merchants |
| **Later** | Multi-touch attribution + discount saturation modeling | Capture full customer journey and cannibalization |
| **Future** | Auto-experiment designer ( Thompson sampling / bandits ) | Self-improving discount policies |

---

## License

Proprietary — this is a hackathon demo. Contact the author before reuse or redistribution.

---

Built for the **Agentic AI** hackathon track. Discount Lens: *stop guessing which discounts work.*
