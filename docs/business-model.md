# Discount Lens — Business Model

**Tagline:** "See the discounts that were never needed."
**Supporting line:** "Turn discount-driven revenue into incremental profit."

---

## 1. Executive Summary

E-commerce and retail companies spend 5–15% of revenue on promotions, yet they measure those promotions with the wrong yardstick — revenue generated, orders placed, conversion rate, and campaign ROI. None of those metrics answer the only question that matters to the P&L:

> **Did the discount actually cause the customer to buy?**

Discount Lens is an agentic AI product that identifies **discount leakage** — money spent on coupons and promotions that did not change a customer's decision. It separates customers into behavioral segments, quantifies the margin being wasted, and generates a new discount-targeting policy with a simulated financial impact. The outcome is not "give fewer discounts"; it is "give discounts where they create incremental value."

---

## 2. The Problem We Solve

### Current state (the pain)
- Promotion budgets are allocated and measured by top-line metrics: revenue, orders, conversion, ROI.
- A customer who would have bought anyway still receives the coupon; that discount is pure margin loss.
- Marketing teams cannot distinguish customers who **need** an incentive from those who **don't**.
- Finance and growth teams cannot see, in dollar terms, how much margin is being given away unnecessarily.

### The four questions Discount Lens answers
1. **Where are we wasting discount?** — identify potentially unnecessary discounts.
2. **Who actually needs a discount?** — identify Persuadables.
3. **How much margin could we recover?** — quantify the economic opportunity.
4. **What should our new discount policy be?** — generate and simulate an optimized targeting rule.

### Why now
- Rising customer acquisition costs force merchants to defend margin, not just grow revenue.
- Agentic AI makes causal analysis of promotion data accessible without a data science team.
- Privacy-safe first-party transaction data is already sitting in every merchant's warehouse, unused.

---

## 3. The Product

### Core value proposition
Discount Lens is an **AI agent**, not a static dashboard. A user hands over transaction, order, or campaign data and the agent:

1. Inspects the dataset and understands its schema
2. Detects available and missing signals
3. Validates data quality
4. Determines what analysis is possible
5. Builds an uplift model to estimate each customer's incremental response to discounts
6. Segments customers (Sure Things, Persuadables, Lost Causes, Price Sensitive)
7. Quantifies discount leakage and recoverable margin
8. Ranks recovery opportunities by financial impact
9. Recommends a new discount policy
10. Simulates the policy's financial impact
11. Prepares deployment-ready rules

### Key features
| Feature | What it does |
|---|---|
| Revenue Agent | Proactive investigation and a Daily Revenue Intelligence Brief |
| Investigation Trail | Visible OBSERVE → VALIDATE → MODEL → CHECK → INVESTIGATE → RECOMMEND workflow |
| Agent Chat | Plain-English questions routed to data tools, with session memory |
| Opportunity Ranking | Scored, deduplicated recovery opportunities across segments, channels, campaigns, products, regions |
| Uplift Segmentation | Model-derived Sure Thing / Persuadable / Price Sensitive / Lost Cause classification |
| Evidence Drill-Down | Transaction-level rows backing every claim |
| Policy Simulator | Side-by-side current vs. proposed financial impact |
| Policy Optimizer | Searches policy combinations for best profit under a revenue-retention constraint |
| Policy Deployment | Versioned, deployment-ready rules |

### The customer segments (uplift model)
- **Sure Things** — high baseline intent, near-zero lift. Discounts here are leakage.
- **Persuadables** — genuine hesitation, positive lift. Discounts here create value.
- **Price Sensitive** — respond to discounts but spend less; needs careful targeting.
- **Lost Causes** — unlikely to convert regardless of incentive.

---

## 4. Customer Segments (Market)

### Target customers
| Segment | Why they buy | Value driver |
|---|---|---|
| **Mid-market e-commerce** (₹50L–₹100Cr GMV) | Margin pressure, no in-house data science | Fast, self-serve insight |
| **DTC / Shopify-native brands** | Promotion-heavy growth, thin margins | Plug-in-able, deployment rules |
| **Growth & retention teams** | Own the discount budget | Attribution they can act on |
| **Finance / FP&A teams** | Own the P&L, want leakage quantified | Recoverable margin in dollars |
| **Agencies & consultants** | Deliver promotion optimization to clients | Repeatable, white-label insight |

### Early adopters / beachhead
Direct-to-consumer brands running frequent promo campaigns (flash sales, welcome codes, retargeting offers) where discount leakage is highest and margin recovery is most visible.

---

## 5. Revenue Model

### Primary: SaaS subscription (per-month, tiered by scale)
| Tier | Target | Price anchor | Includes |
|---|---|---|---|
| **Starter** | SMB / early DTC | $499/mo | Up to 500k transactions/mo, monthly analysis, 3 seats |
| **Growth** | Mid-market | $1,499/mo | Up to 5M transactions/mo, weekly agent runs, policy optimization |
| **Scale** | Enterprise / high-GMV | Custom | Unlimited data, API access, deployment integrations, SSO, SLA |

### Secondary: usage-based / value-based components
- **Recovered-margin success fee** — a percentage of margin actually recovered (performance-linked, aligns incentives).
- **Deployment integrations** — connectors to promotion engines, CDPs, and ESPs (Klaviyo, Shopify, Braze, Iterable) as add-ons.
- **Agency / white-label licenses** — per-client pricing for consultants.

### Why this works
- The ROI is directly measurable: if a merchant recovers ₹1.8L/month in margin, a ₹1.5K/month subscription is a ~100x return.
- Value-based pricing creates an anchor that scales with customer success, not seat count.

---

## 6. Unit Economics (illustrative)

| Metric | Starter | Growth |
|---|---|---|
| Monthly price | $499 | $1,499 |
| Cost to serve (compute + LLM + support) | ~$60 | ~$250 |
| Gross margin | ~88% | ~83% |
| CAC (sales-led + self-serve blend) | $1,500 | $6,000 |
| Payback period | ~3 months | ~4 months |
| Annual LTV (24-month retention) | ~$9,000 | ~$27,000 |

- **Retention driver:** the product compounds — each new campaign run surfaces fresh leakage, so the value recurs monthly.
- **Expansion driver:** more data → more leakage found → tier upgrade.

---

## 7. Go-To-Market

### Motion: Product-led with a sales-assisted enterprise layer
1. **Free value upfront** — a free diagnostic that uploads a sample / first dataset and shows the recoverable-margin opportunity (the "leakage teaser").
2. **Self-serve onboarding** — upload CSV, agent runs, see the Daily Brief within minutes.
3. **Sales-assisted conversion** — for accounts above a GMV threshold, a value consultant quantifies the opportunity and closes on ROI.
4. **Land-and-expand** — start with the growth team, expand to finance, then enterprise deployment.

### Channels
- Content & thought leadership on promotion attribution ("your discount ROI is lying to you").
- Partnerships with e-commerce platforms and agencies.
- Founder-led demo to early DTC communities and hackathon/incubator networks.

### Distribution of the hackathon demo
- The demo runs on a single Flask app with a seeded 2,500-row causal dataset — a self-contained proof that the agent investigates, segments, simulates, and deploys without external dependencies.

---

## 8. Competition & Differentiation

| Alternative | What they measure | What they miss |
|---|---|---|
| Shopify / Klaviyo native analytics | Revenue, orders, conversion | Whether the discount **caused** the purchase |
| Attribution tools | Click-path attribution | Counterfactual uplift ("would they have bought anyway?") |
| BI dashboards | Descriptive metrics | The **action** — a recommended policy |
| Manual data science | Custom uplift models | Cost, time, repeatability, no agent |

### Discount Lens moats
1. **Causal uplift, not attribution** — the intellectual core that answers the counterfactual.
2. **Agentic workflow** — visible investigation, plain-English Q&A, and a recommendation, not just a chart.
3. **Actionability** — it ends in a simulated, deployment-ready policy, closing the loop to business value.

---

## 9. Key Metrics (North Star)

- **Recoverable margin identified** (the value we surface) — the leading indicator of product value.
- **Net revenue retention (NRR)** — target >120% via land-and-expand.
- **Time-to-value** — minutes from upload to first Brief.
- **Activation rate** — % of signups who run the agent and view an opportunity.
- **Deployment rate** — % who simulate or deploy a recommended policy (the moment value is realized).

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Customers distrust "causal" claims | Lead with honesty — reliability grades, observational framing, evidence rows |
| Requires clean transaction data | Schema inspection, missing-signal detection, guided data prep |
| LLM/inference cost at scale | Batch causal estimation; LLM only for the agent layer, not the math |
| Churn if leakage is "fixed" once | Recurring monitoring as new campaigns ship; expanding data scope |
| Competitors copying the feature | Compounding data + workflow moat; fast, opinionated agent UX |

---

## 11. Roadmap

- **Phase 1 (now)** — single-tenant demo: upload → agent → brief → simulate → deploy.
- **Phase 2** — connectors to promotion engines (Shopify, Klaviyo), scheduled agent runs.
- **Phase 3** — multi-tenant SaaS, team seats, and value-based billing on recovered margin.
- **Phase 4** — proactive alerts and autonomous policy enforcement with human approval gates.

---

## 12. The Ask / Funding Use

For a hackathon or seed context, the pitch closes on three points:

1. **The insight is differentiated** — attribution tells you what happened; Discount Lens tells you what *would* have happened.
2. **The value is measurable** — every opportunity is priced in recoverable margin, so ROI is provable in the demo.
3. **The product is agentic end-to-end** — from raw data to a deployment-ready policy in one workflow, which is exactly what "agentic commerce" should mean.
