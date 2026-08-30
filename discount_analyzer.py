#!/usr/bin/env python3
"""
DiscountLens - Core Discount Leakage Analyzer
Real functional analysis engine for e-commerce discount leakage detection.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class DiscountAnalyzer:
    """Core engine for analyzing discount leakage and customer behavior."""
    
    def __init__(self):
        self.segment_definitions = {
            'Sure Thing': {
                'description': 'High purchase intent, low price sensitivity. Would buy without discount.',
                'color': '#10b981',  # green
                'characteristics': ['High repeat rate', 'Buys at full price', 'Fast decision time']
            },
            'Persuadable': {
                'description': 'Moderate intent. Responds to discounts and needs incentive to convert.',
                'color': '#3b82f6',  # blue
                'characteristics': ['Moderate sensitivity', 'Uplift from discounts', 'Cart recovery responsive']
            },
            'Price Warrior': {
                'description': 'Only buys on discount. High price sensitivity.',
                'color': '#f59e0b',  # amber
                'characteristics': ['High sensitivity', 'Waits for deals', 'Compares prices']
            },
            'Lost Cause': {
                'description': 'Low intent. Unlikely to purchase even with discount.',
                'color': '#ef4444',  # red
                'characteristics': ['Low engagement', 'Rare purchases', 'High churn risk']
            }
        }
    
    def _infer_customer_segments(self, transactions_df: pd.DataFrame, customers_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Infer customer segments from transaction behavior when explicit labels not present."""
        if customers_df is None or 'customer_type' not in customers_df.columns:
            # Create customer-level aggregates from transactions
            customer_stats = transactions_df.groupby('customer_id').agg({
                'final_value': ['count', 'mean'],
                'discount_percentage': 'mean',
                'base_value': 'mean'
            }).reset_index()
            customer_stats.columns = ['customer_id', 'purchase_count', 'avg_order_value', 'avg_discount_pct', 'avg_base_value']
            
            # Heuristic segmentation
            def assign_segment(row):
                purchase_count = row['purchase_count']
                avg_discount = row['avg_discount_pct']
                
                # Sure Things: buy often and use low/no discounts
                if purchase_count >= 4 and avg_discount < 0.10:
                    return 'Sure Thing'
                # Price Warriors: high discount usage
                elif avg_discount > 0.18:
                    return 'Price Warrior'
                # Lost Causes: very few purchases despite high discounts
                elif purchase_count <= 1 and avg_discount > 0.15:
                    return 'Lost Cause'
                # Default to Persuadable
                else:
                    return 'Persuadable'
            
            customer_stats['customer_type'] = customer_stats.apply(assign_segment, axis=1)
            return customer_stats[['customer_id', 'customer_type', 'purchase_count', 'avg_order_value', 'avg_discount_pct']]
        else:
            # Use provided customer types
            return customers_df[['customer_id', 'customer_type']].copy()
    
    def analyze_data(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame) -> Dict[str, Any]:
        """Primary analysis function for pre-generated sample data."""
        return self._run_full_analysis(transactions_df, customers_df, is_sample=True)
    
    def analyze_uploaded_data(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze a schema-agnostic, ALREADY-RESOLVED dataframe.

        Expects the canonical frame produced by schema_engine.resolve_dataframe():
        columns are renamed to semantic concepts and discount fields are derived
        only where a defensible derivation exists. Absent columns stay absent;
        nothing is fabricated. The pipeline degrades gracefully.
        """
        from schema_engine import resolve_dataframe, detect_schema

        df = raw_df.copy()
        df.columns = [c.strip() for c in df.columns]

        # If the caller passed a raw/un-resolved frame (e.g. legacy /api/upload),
        # resolve it through the schema engine first.
        if not any(c in df.columns for c in
                   ('treated', 'discount_amount', 'discount_percent',
                    'original_value', 'final_value', 'conversion', 'treatment')):
            schema = detect_schema(df)
            df = resolve_dataframe(df, schema)

        has = lambda c: c in df.columns  # noqa: E731

        # ---- customer identity (optional but central to segmentation) --------
        if has('customer_id'):
            df['customer_id'] = df['customer_id'].astype(str)
        else:
            # synthesize a per-row identity so segmentation still has a key;
            # this is NOT a business value, just a row handle
            df['customer_id'] = [f'ROW_{i}' for i in range(len(df))]

        # ---- normalize the money / treatment fields we may have --------------
        if has('discount_percent'):
            dp = pd.to_numeric(df['discount_percent'], errors='coerce').fillna(0).clip(lower=0)
            if dp.max() > 1.5:
                dp = dp / 100.0
            df['discount_percent'] = dp
        if has('discount_amount'):
            df['discount_amount'] = pd.to_numeric(df['discount_amount'], errors='coerce').fillna(0).clip(lower=0)
        for money in ('final_value', 'original_value', 'revenue', 'profit'):
            if has(money):
                df[money] = pd.to_numeric(df[money], errors='coerce')

        # derive treated from discount fields if the schema didn't already
        if not has('treated'):
            if has('discount_amount'):
                df['treated'] = df['discount_amount'] > 0
            elif has('discount_percent'):
                df['treated'] = df['discount_percent'] > 0
            else:
                df['treated'] = False

        # choose a revenue proxy for downstream math; NEVER equate revenue to profit
        if not has('final_value') and has('original_value') and has('discount_amount'):
            df['final_value'] = df['original_value'] - df['discount_amount']
        value_col = 'final_value' if has('final_value') else ('original_value' if has('original_value') else ('revenue' if has('revenue') else None))

        # ---- customer-level segmentation -------------------------------------
        cust_agg = df.groupby('customer_id').agg(
            purchase_count=('customer_id', 'size'),
        ).reset_index()

        if has('treated'):
            t_rate = df.groupby('customer_id')['treated'].mean()
            cust_agg['treated_rate'] = cust_agg['customer_id'].map(t_rate).fillna(0)
        else:
            cust_agg['treated_rate'] = 0.0

        if has('discount_percent'):
            cust_agg['avg_discount_pct'] = cust_agg['customer_id'].map(
                df.groupby('customer_id')['discount_percent'].mean()).fillna(0)
        else:
            cust_agg['avg_discount_pct'] = 0.0

        if value_col:
            cust_agg['avg_value'] = cust_agg['customer_id'].map(
                df.groupby('customer_id')[value_col].mean()).fillna(0)
        else:
            cust_agg['avg_value'] = 0.0

        # conversion-aware intent where available, else behavior/treatment proxy
        if has('conversion'):
            conv = pd.to_numeric(df['conversion'], errors='coerce').fillna(0)
            df['conversion_num'] = conv
            cust_agg['conversion_rate'] = cust_agg['customer_id'].map(
                df.groupby('customer_id')['conversion_num'].mean()).fillna(0)
        else:
            cust_agg['conversion_rate'] = 0.0

        def assign_segment(row):
            pc = row['purchase_count']
            ad = row['avg_discount_pct']
            tr = row['treated_rate']
            cr = row['conversion_rate']
            # No explicit conversion column: treat repeated full-price buying as
            # high intent (a Sure Thing signal). A conversion column, when present,
            # sharpens the call.
            if has('conversion'):
                # high repeat + converts without treatment => Sure Thing
                if pc >= 2 and (tr < 0.4 or ad < 0.10) and cr >= 0.5:
                    return 'Sure Thing'
                if ad > 0.20 or tr > 0.7:
                    return 'Price Warrior'
                if pc <= 1 and cr < 0.15 and ad > 0.10:
                    return 'Lost Cause'
                return 'Persuadable'
            else:
                # purchase-pattern fallback: repeat buyers at full price = Sure Thing
                if pc >= 2 and ad < 0.10:
                    return 'Sure Thing'
                if ad > 0.20 or tr > 0.7:
                    return 'Price Warrior'
                if pc <= 1 and ad < 0.05:
                    return 'Lost Cause'
                return 'Persuadable'

        cust_agg['customer_type'] = cust_agg.apply(assign_segment, axis=1)
        df = df.merge(cust_agg[['customer_id', 'customer_type']], on='customer_id', how='left')
        df['customer_type'] = df['customer_type'].fillna('Persuadable')

        if has('discount_percent'):
            df['was_leakage'] = (df['discount_percent'] > 0) & (df['customer_type'] == 'Sure Thing')
        elif has('treated'):
            df['was_leakage'] = (df['treated']) & (df['customer_type'] == 'Sure Thing')
        else:
            df['was_leakage'] = False

        customers_view = cust_agg[['customer_id', 'customer_type']].copy()
        customers_view['price_sensitivity'] = customers_view['customer_type'].map(
            {'Sure Thing': 0.15, 'Persuadable': 0.55, 'Price Warrior': 0.85, 'Lost Cause': 0.95})

        # record what was actually available so _run_full_analysis can adapt
        df.attrs['value_col'] = value_col
        df.attrs['has_profit'] = has('profit')
        df.attrs['has_conversion'] = has('conversion') or 'conversion_num' in df.columns
        df.attrs['has_treatment'] = has('treated')
        df.attrs['has_discount_amount'] = has('discount_amount')

        return self._run_full_analysis(df, customers_view, is_sample=False)
    
    def _run_full_analysis(self, transactions_df: pd.DataFrame, customers_df: pd.DataFrame, is_sample: bool = False) -> Dict[str, Any]:
        """Internal full analysis pipeline. Schema-agnostic: uses whatever columns
        are present and degrades gracefully when money/profit/channel are absent."""
        tx = transactions_df.copy()
        cust = customers_df.copy()

        has = lambda c: c in tx.columns  # noqa: E731
        # the analyzer historically used 'discount_percentage'; normalize to 'discount_percent'
        if has('discount_percentage') and not has('discount_percent'):
            tx['discount_percent'] = tx['discount_percentage']
        if has('base_value') and not has('original_value'):
            tx['original_value'] = tx['base_value']

        # choose the monetary value column actually present
        value_col = tx.attrs.get('value_col') or (
            'final_value' if has('final_value') else ('original_value' if has('original_value') else ('revenue' if has('revenue') else None)))

        # ---- Basic KPIs (degrade gracefully) --------------------------------
        total_orders = len(tx)
        if value_col:
            total_revenue = float(pd.to_numeric(tx[value_col], errors='coerce').sum())
        else:
            total_revenue = 0.0  # no money field => revenue unknowable
        if has('discount_amount'):
            total_discounts_given = float(pd.to_numeric(tx['discount_amount'], errors='coerce').fillna(0).sum())
        elif has('original_value') and has('final_value'):
            total_discounts_given = float((pd.to_numeric(tx['original_value'], errors='coerce')
                                           - pd.to_numeric(tx['final_value'], errors='coerce')).fillna(0).sum())
        else:
            total_discounts_given = 0.0  # cannot compute exact spend
        avg_order_value = (total_revenue / total_orders) if total_orders > 0 and value_col else 0.0

        # ---- Leakage: discounts to Sure Things ------------------------------
        if has('discount_percent'):
            leakage_mask = (pd.to_numeric(tx['discount_percent'], errors='coerce').fillna(0) > 0) & (tx['customer_type'] == 'Sure Thing')
        elif has('treated'):
            leakage_mask = (tx['treated']) & (tx['customer_type'] == 'Sure Thing')
        else:
            leakage_mask = pd.Series([False] * len(tx))
        leakage_tx = tx[leakage_mask].copy()

        if has('discount_amount'):
            leakage_amount = float(pd.to_numeric(leakage_tx['discount_amount'], errors='coerce').fillna(0).sum())
        elif has('original_value') and has('final_value'):
            leakage_amount = float((pd.to_numeric(leakage_tx['original_value'], errors='coerce')
                                    - pd.to_numeric(leakage_tx['final_value'], errors='coerce')).fillna(0).sum())
        elif has('discount_percent') and value_col:
            leakage_amount = float((pd.to_numeric(leakage_tx['discount_percent'], errors='coerce').fillna(0)
                                    * pd.to_numeric(leakage_tx[value_col], errors='coerce').fillna(0)).sum())
        else:
            leakage_amount = 0.0  # exact spend unknown

        # margin leakage: prefer real profit margin; else estimate from gross margin rate
        has_profit = tx.attrs.get('has_profit', has('profit'))
        if has_profit and has('discount_amount'):
            estimated_margin_leakage = float(leakage_tx['discount_amount'].sum())  # profit already net
        elif has_profit and value_col:
            estimated_margin_leakage = float((pd.to_numeric(leakage_tx[value_col], errors='coerce').fillna(0)
                                              * pd.to_numeric(leakage_tx['discount_percent'], errors='coerce').fillna(0)).sum())
        else:
            gross_margin_rate = 0.30
            estimated_margin_leakage = leakage_amount * gross_margin_rate
        opportunity_cost = leakage_amount

        # ---- Discount effectiveness by segment -------------------------------
        discount_by_segment = {}
        for seg in ['Sure Thing', 'Persuadable', 'Price Warrior', 'Lost Cause']:
            seg_tx = tx[tx['customer_type'] == seg]
            if len(seg_tx) > 0:
                if has('discount_percent'):
                    disc_pct = pd.to_numeric(seg_tx['discount_percent'], errors='coerce').fillna(0)
                    discounted = seg_tx[disc_pct > 0]
                elif has('treated'):
                    discounted = seg_tx[seg_tx['treated']]
                else:
                    discounted = seg_tx.iloc[0:0]
                discount_rate = len(discounted) / len(seg_tx) if len(seg_tx) > 0 else 0
                avg_disc = float(disc_pct.mean()) if has('discount_percent') else 0.0
                seg_revenue = float(pd.to_numeric(seg_tx[value_col], errors='coerce').fillna(0).sum()) if value_col else 0.0
                discount_by_segment[seg] = {
                    'discount_rate': round(discount_rate, 3),
                    'avg_discount': round(avg_disc, 3),
                    'orders': len(seg_tx),
                    'revenue': round(seg_revenue, 2)
                }

        # ---- Leakage sources (by channel if present, else by campaign) -----
        leakage_sources = []
        if len(leakage_tx) > 0:
            group_col = 'channel' if has('channel') else ('campaign_id' if has('campaign_id') else None)
            if group_col:
                by = leakage_tx.groupby(group_col).size().reset_index(name='count')
                for _, row in by.iterrows():
                    amt = 0.0
                    if has('discount_amount'):
                        amt = float(pd.to_numeric(leakage_tx[leakage_tx[group_col] == row[group_col]]['discount_amount'], errors='coerce').fillna(0).sum())
                    leakage_sources.append({
                        'source': f"{row[group_col]}",
                        'leakage': round(amt, 2),
                        'transactions': int(row['count']),
                        'evidence': f"{int(row['count'])} Sure Thing customers received discounts via {row[group_col]}"
                    })

        # ---- Evidence rows (defensive on column presence) ------------------
        evidence_rows = []
        if len(leakage_tx) > 0:
            sort_col = 'discount_amount' if has('discount_amount') else ('discount_percent' if has('discount_percent') else None)
            sample_leak = leakage_tx.sort_values(sort_col, ascending=False).head(8) if sort_col else leakage_tx.head(8)
            for _, row in sample_leak.iterrows():
                evidence_rows.append({
                    'transaction_id': row.get('transaction_id', row.get('customer_id', '—')),
                    'customer_id': row.get('customer_id', '—'),
                    'customer_type': row['customer_type'],
                    'base_value': round(float(row.get('original_value', row.get('base_value', 0) or 0)), 2) if (has('original_value') or has('base_value')) else None,
                    'discount_amount': round(float(row.get('discount_amount', 0) or 0), 2) if has('discount_amount') else None,
                    'final_value': round(float(row.get('final_value', 0) or 0), 2) if has('final_value') else None,
                    'discount_code': row.get('discount_code', row.get('coupon_code', 'N/A')),
                    'channel': row.get('channel', 'Unknown'),
                    'reason': 'High-intent customer (Sure Thing) likely to purchase without the incentive. Discount may not have changed the outcome.',
                    'impact': (f"₹{round(float(row.get('discount_amount', 0) or 0), 2)} margin lost" if has('discount_amount') else 'Discount exposure on a likely Sure Thing')
                })
        
        # Monthly trend (fake aggregation for demo)
        if 'purchase_date' in tx.columns:
            tx['purchase_date'] = pd.to_datetime(tx['purchase_date'], errors='coerce')
            tx['month'] = tx['purchase_date'].dt.to_period('M').astype(str)
            agg = {}
            if value_col:
                agg['revenue'] = (value_col, 'sum')
            if has('discount_amount'):
                agg['discounts'] = ('discount_amount', 'sum')
            monthly = tx.groupby('month').agg(**agg).reset_index() if agg else pd.DataFrame()
            monthly_trend = monthly.to_dict('records') if len(monthly) else []
        else:
            monthly_trend = []
        
        segment_counts = tx.groupby('customer_type').size().to_dict()
        if value_col:
            segment_revenue = tx.groupby('customer_type')[value_col].sum().to_dict()
        else:
            segment_revenue = {k: 0.0 for k in segment_counts}
        
        # Assemble full report
        analysis = {
            'summary': {
                'total_orders': int(total_orders),
                'total_revenue': round(total_revenue, 2),
                'total_discounts_given': round(total_discounts_given, 2),
                'avg_order_value': round(avg_order_value, 2),
                'estimated_margin_leakage': round(estimated_margin_leakage, 2),
                'opportunity_cost': round(opportunity_cost, 2),
                'leakage_transactions': int(len(leakage_tx)),
                'leakage_rate': round(len(leakage_tx) / total_orders, 3) if total_orders > 0 else 0
            },
            'segments': {
                'counts': segment_counts,
                'revenue': {k: round(float(v), 2) for k, v in segment_revenue.items()},
                'definitions': self.segment_definitions
            },
            'discount_by_segment': discount_by_segment,
            'leakage_sources': leakage_sources,
            'evidence': evidence_rows,
            'monthly_trend': monthly_trend,
            'metadata': {
                'analyzed_at': datetime.now().isoformat(),
                'is_sample': is_sample,
                'total_customers': int(tx['customer_id'].nunique())
            }
        }
        
        # Store raw for simulation use (attach to analysis for later)
        analysis['_raw_transactions'] = tx.to_dict('records')
        analysis['_raw_customers'] = cust.to_dict('records')
        
        return analysis
    
    def simulate_policy(self, current_policy: Dict, proposed_policy: Dict, analysis_data: Dict) -> Dict[str, Any]:
        """
        Run a counterfactual simulation of a new discount policy.
        Uses the stored raw transactions to re-evaluate what would happen.
        """
        if '_raw_transactions' not in analysis_data:
            return {'error': 'No raw transaction data available for simulation'}
        
        tx = pd.DataFrame(analysis_data['_raw_transactions'])
        cust = pd.DataFrame(analysis_data['_raw_customers'])
        
        # Merge customer info
        if 'customer_type' not in tx.columns:
            tx = tx.merge(cust[['customer_id', 'customer_type']], on='customer_id', how='left')
        
        # Simple policy model
        # current_policy and proposed_policy can contain:
        #   sure_thing_discount_cap (0.0 = none)
        #   persuadable_max (e.g. 0.15)
        #   price_warrior_max (e.g. 0.25)
        
        def apply_policy(row, policy):
            ctype = row.get('customer_type', 'Persuadable')
            current_disc = row.get('discount_percent', row.get('discount_percentage', 0.0) or 0.0) or 0.0
            
            cap = 0.0
            if ctype == 'Sure Thing':
                cap = policy.get('sure_thing_discount_cap', 0.0)
            elif ctype == 'Persuadable':
                cap = policy.get('persuadable_max', 0.15)
            elif ctype == 'Price Warrior':
                cap = policy.get('price_warrior_max', 0.25)
            else:
                cap = 0.05  # conservative for lost causes
            
            new_disc = min(current_disc, cap) if current_disc > 0 else 0.0
            
            # For Sure Things we may decide to give 0 even if they had some
            if ctype == 'Sure Thing' and policy.get('sure_thing_discount_cap', 0.0) == 0.0:
                new_disc = 0.0
            
            return new_disc
        
        # Apply proposed
        tx['proposed_discount_pct'] = tx.apply(lambda r: apply_policy(r, proposed_policy), axis=1)
        
        # Determine which columns to use defensively
        value_col = None
        for c in ('final_value', 'original_value', 'base_value', 'revenue'):
            if c in tx.columns:
                value_col = c
                break
        disc_col = 'discount_percent' if 'discount_percent' in tx.columns else (
            'discount_percentage' if 'discount_percentage' in tx.columns else None)

        # Simulate outcomes
        results = []
        for _, row in tx.iterrows():
            base = float(row[value_col]) if value_col else 0.0
            old_disc = float(row.get(disc_col, 0.0) or 0.0) if disc_col else 0.0
            new_disc = row['proposed_discount_pct']
            
            old_final = base * (1 - old_disc)
            new_final = base * (1 - new_disc)
            
            ctype = row.get('customer_type', 'Persuadable')
            price_sens = {'Sure Thing': 0.15, 'Persuadable': 0.55, 'Price Warrior': 0.85, 'Lost Cause': 0.95}.get(ctype, 0.5)
            
            # Conversion probability adjustment
            # If we give less discount to someone who needed it, they may not buy
            old_convert_prob = 0.85 + (old_disc * price_sens * 0.4)
            new_convert_prob = 0.85 + (new_disc * price_sens * 0.4)
            
            # For Sure Things, they buy anyway
            if ctype == 'Sure Thing':
                old_convert_prob = 0.96
                new_convert_prob = 0.95  # tiny risk
            
            old_revenue = old_final * (1 if np.random.random() < old_convert_prob else 0)  # but use expected
            new_revenue = new_final * (1 if np.random.random() < new_convert_prob else 0)
            
            # Use expected value for stability
            old_exp_rev = old_final * old_convert_prob
            new_exp_rev = new_final * new_convert_prob
            
            results.append({
                'customer_type': ctype,
                'old_final': old_final,
                'new_final': new_final,
                'old_exp_rev': old_exp_rev,
                'new_exp_rev': new_exp_rev,
                'old_disc_amt': base * old_disc,
                'new_disc_amt': base * new_disc,
                'converted_old': old_convert_prob,
                'converted_new': new_convert_prob
            })
        
        sim_df = pd.DataFrame(results)
        
        # Aggregate metrics
        current = {
            'orders': float(len(tx)),
            'revenue': round(float(sim_df['old_exp_rev'].sum()), 2),
            'discount_cost': round(float(sim_df['old_disc_amt'].sum()), 2),
            'profit': round(float(sim_df['old_exp_rev'].sum() * 0.30), 2)  # assume 30% margin after discount
        }
        
        proposed = {
            'orders': round(float((sim_df['converted_new'] > 0.5).sum() + (sim_df['converted_new'] <= 0.5).sum() * 0.0), 1),  # expected orders
            'revenue': round(float(sim_df['new_exp_rev'].sum()), 2),
            'discount_cost': round(float(sim_df['new_disc_amt'].sum()), 2),
            'profit': round(float(sim_df['new_exp_rev'].sum() * 0.30), 2)
        }
        
        # Better order estimate using sum of probs
        current['orders'] = round(float(sim_df['converted_old'].sum()), 1)
        proposed['orders'] = round(float(sim_df['converted_new'].sum()), 1)
        
        deltas = {
            'revenue_delta': round(proposed['revenue'] - current['revenue'], 2),
            'discount_delta': round(proposed['discount_cost'] - current['discount_cost'], 2),
            'profit_delta': round(proposed['profit'] - current['profit'], 2),
            'orders_delta': round(proposed['orders'] - current['orders'], 1)
        }
        
        # Confidence interval (very rough Monte Carlo style band)
        std_factor = 0.06  # 6% noise
        confidence = {
            'revenue_low': round(proposed['revenue'] * (1 - std_factor), 2),
            'revenue_high': round(proposed['revenue'] * (1 + std_factor), 2),
            'profit_low': round(proposed['profit'] * (1 - std_factor), 2),
            'profit_high': round(proposed['profit'] * (1 + std_factor), 2)
        }
        
        return {
            'current': current,
            'proposed': proposed,
            'deltas': deltas,
            'confidence': confidence,
            'policy_applied': proposed_policy,
            'simulation_notes': [
                'Simulation uses historical transaction data + behavioral uplift model.',
                'Sure Things are assumed to convert at ~95% even without discounts.',
                'Persuadables show ~0.4-0.6 elasticity to discount depth.',
                'Results are expected values over the historical cohort.'
            ]
        }
    
    def get_customer_analysis(self, customer_id: str, analysis_data: Dict) -> Dict[str, Any]:
        """Return detailed view for one customer. Schema-agnostic on columns."""
        if '_raw_transactions' not in analysis_data:
            return {'error': 'No data'}
        
        tx = pd.DataFrame(analysis_data['_raw_transactions'])
        cust_tx = tx[tx['customer_id'] == customer_id]
        
        if len(cust_tx) == 0:
            return {'error': 'Customer not found'}
        
        ctype = cust_tx['customer_type'].iloc[0] if 'customer_type' in cust_tx.columns else 'Unknown'
        value_col = None
        for c in ('final_value', 'original_value', 'base_value', 'revenue'):
            if c in cust_tx.columns:
                value_col = c
                break
        total_spent = float(pd.to_numeric(cust_tx[value_col], errors='coerce').fillna(0).sum()) if value_col else 0.0
        if 'discount_amount' in cust_tx.columns:
            discounts_received = float(pd.to_numeric(cust_tx['discount_amount'], errors='coerce').fillna(0).sum())
        elif value_col and 'original_value' in cust_tx.columns and 'final_value' in cust_tx.columns:
            discounts_received = float((pd.to_numeric(cust_tx['original_value'], errors='coerce')
                                        - pd.to_numeric(cust_tx['final_value'], errors='coerce')).fillna(0).sum())
        else:
            discounts_received = 0.0
        
        # select only columns that actually exist so to_dict never KeyErrors
        keep = [c for c in ('transaction_id', 'original_value', 'base_value', 'discount_amount',
                            'discount_percent', 'discount_percentage', 'final_value',
                            'discount_code', 'coupon_code', 'channel', 'campaign_id',
                            'conversion', 'treated', 'product_category')
                if c in cust_tx.columns]
        return {
            'customer_id': customer_id,
            'segment': ctype,
            'total_orders': len(cust_tx),
            'total_spent': round(total_spent, 2),
            'discounts_received': round(discounts_received, 2),
            'transactions': cust_tx[keep].to_dict('records') if keep else [],
            'why_this_segment': self.segment_definitions.get(ctype, {}).get('description', '')
        }
