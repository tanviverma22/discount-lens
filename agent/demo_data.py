"""Programmatic causal-ready demo data for Discount Lens."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_demo_dataset(rows: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate observational commerce data with known, realistic uplift patterns.

    The latent intent and treatment response are generated first; conversion is then
    sampled. No dashboard value is embedded in this generator.
    """
    rng = np.random.default_rng(seed)
    customer_ids = rng.choice([f"CUST_{i:04d}" for i in range(max(400, rows // 3))], rows)
    channel = rng.choice(["Email", "PPC", "Organic", "Social", "Direct"], rows,
                         p=[.24, .22, .20, .18, .16])
    campaign = rng.choice(["WELCOME10", "SUMMER15", "FLASH20", "LOYALTY", "RETARGET"], rows)
    category = rng.choice(["Beauty", "Electronics", "Fashion", "Home", "Sports"], rows)
    region = rng.choice(["North", "South", "East", "West"], rows)

    returning = rng.binomial(1, .46, rows)
    product_views = rng.poisson(4.5 + 2.4 * returning, rows).clip(1, 20)
    cart_added = rng.binomial(1, np.clip(.13 + .055 * product_views, .1, .86))
    dwell_time_seconds = np.clip(rng.gamma(3, 70, rows) + 50 * cart_added, 20, 1800).round().astype(int)
    price_comparison_tabs = rng.poisson(.7 + .9 * (1 - returning), rows).clip(0, 8)
    prior_orders = rng.poisson(.7 + 2.2 * returning, rows).clip(0, 12)

    # Pre-treatment baseline purchase propensity.
    baseline_logit = (-3.1 + .16 * product_views + 1.05 * cart_added
                      + .0010 * dwell_time_seconds - .24 * price_comparison_tabs
                      + .28 * returning + .17 * prior_orders)
    baseline_probability = 1 / (1 + np.exp(-baseline_logit))

    # Treatment assignment is observational: target lower baseline and retarget/email.
    treatment_logit = (-.35 + .55 * (channel == "Email") + .44 * (channel == "PPC")
                       + .35 * (campaign == "RETARGET") + .28 * price_comparison_tabs
                       - .55 * baseline_probability)
    propensity = 1 / (1 + np.exp(-treatment_logit))
    treated = rng.binomial(1, propensity)

    # True lift is biggest for hesitant, price-comparing visitors, and near zero for high intent.
    treatment_lift = np.clip(.23 * (1 - baseline_probability) + .026 * price_comparison_tabs
                             - .06 * returning - .025 * prior_orders, -.025, .30)
    discount_percentage = np.where(treated, rng.choice([.05, .10, .15, .20, .25], rows,
                                                          p=[.13, .28, .30, .19, .10]), 0.0)
    order_value = np.clip(rng.lognormal(4.72, .45, rows), 25, 650).round(2)
    discount_amount = (order_value * discount_percentage).round(2)
    conversion_probability = np.clip(baseline_probability + treated * treatment_lift, .01, .985)
    converted = rng.binomial(1, conversion_probability)

    final_value = np.where(converted, order_value - discount_amount, 0.0).round(2)
    unit_cost = (order_value * rng.uniform(.45, .68, rows)).round(2)
    profit = np.where(converted, final_value - unit_cost, 0.0).round(2)

    return pd.DataFrame({
        "transaction_id": [f"DEMO_{i + 1:06d}" for i in range(rows)],
        "customer_id": customer_ids,
        "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h").astype(str),
        "channel": channel,
        "campaign_id": campaign,
        "product_category": category,
        "region": region,
        "returning_customer": returning,
        "prior_orders": prior_orders,
        "product_views": product_views,
        "cart_added": cart_added,
        "dwell_time_seconds": dwell_time_seconds,
        "price_comparison_tabs": price_comparison_tabs,
        "original_value": order_value,
        "discount_percent": discount_percentage,
        "discount_amount": discount_amount,
        "treated": treated,
        "conversion": converted,
        "final_value": final_value,
        "cost": unit_cost,
        "profit": profit,
    })
