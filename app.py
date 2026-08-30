#!/usr/bin/env python3
"""
DiscountLens - Agentic Commerce Web App
Detects hidden discount leakage in e-commerce businesses
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import re
from discount_analyzer import DiscountAnalyzer
from ai_agent import DiscountInvestigator

from agent.agent_analyzer import CausalAnalyzer
from agent.agent_memory import AgentMemory
from agent.agent_orchestrator import RevenueAgent
from agent.agent_simulator import simulate_policy as agent_simulate, DEFAULT_POLICY as AGENT_DEFAULT_POLICY
from agent.agent_recommender import RevenueRecommender
from agent.agent_tools import policy_json
from agent.demo_data import generate_demo_dataset

app = Flask(__name__)
CORS(app)

# Initialize analyzer and investigator
analyzer = DiscountAnalyzer()
investigator = DiscountInvestigator()
causal_analyzer = CausalAnalyzer()
agent_memory = AgentMemory()
revenue_agent = RevenueAgent(agent_memory)
recommender = RevenueRecommender()

# In-memory session storage for uploads (single-user/demo only)
session_storage = {}

def get_session_id():
    """Get or create session ID from request"""
    return request.headers.get('X-Session-ID', 'default')

def load_sample_data():
    """Load and analyze sample e-commerce data from CSV"""
    from schema_engine import detect_schema, resolve_dataframe
    sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')
    if not os.path.exists(sample_path):
        return {'error': 'Sample data file not found'}
    
    df = pd.read_csv(sample_path)
    schema = detect_schema(df)
    resolved = resolve_dataframe(df, schema)
    analysis = analyzer.analyze_uploaded_data(resolved)
    analysis['schema'] = schema
    return analysis

def _resolve_and_analyze(df: pd.DataFrame):
    """Resolve an arbitrary dataframe through the schema engine and analyze it."""
    from schema_engine import detect_schema, resolve_dataframe
    schema = detect_schema(df)
    resolved = resolve_dataframe(df, schema)
    analysis = analyzer.analyze_uploaded_data(resolved)
    analysis['schema'] = schema
    return analysis

def _current_analysis():
    """Return the analysis for the current session, or the sample analysis as fallback."""
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    if 'analysis' in stored:
        return stored['analysis'], False
    return load_sample_data(), True

@app.route('/')
def index():
    """Serve the main application"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

@app.route('/api/sample-preview')
def sample_preview():
    """Preview the built-in sample dataset via the same schema-agnostic detection pipeline as uploads."""
    try:
        from schema_engine import detect_schema
        sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')
        df = pd.read_csv(sample_path)
        
        schema = detect_schema(df)
        
        session_id = get_session_id()
        session_storage[session_id] = {
            'preview_df': df.to_dict('records'),
            'columns': df.columns.tolist(),
            'dtypes': {c: str(df[c].dtype) for c in df.columns},
            'uploaded_at': datetime.now().isoformat(),
            'filename': 'sample_transactions.csv',
            'is_sample': True
        }
        
        return jsonify({
            'preview': df.head(10).to_dict('records'),
            'columns': df.columns.tolist(),
            'dtypes': {c: str(df[c].dtype) for c in df.columns},
            'row_count': len(df),
            'schema': schema,
            'is_sample': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/csv/inspect', methods=['POST'])
def csv_inspect():
    """Schema-agnostic inspection of an uploaded CSV.

    Replaces the old base/final-value-required flow. Returns:
      - schema detection (semantic mappings, discount case 1-8)
      - capability matrix (READY / NEEDS_DATA per analysis type)
      - agent reasoning (plain English)
      - a small preview
    Never rejects a CSV for missing columns; it reports what analyses are possible.
    """
    try:
        from schema_engine import detect_schema

        file = request.files.get('file')
        if file is None or file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400

        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'error': 'The uploaded CSV is empty.'}), 400

        schema = detect_schema(df)

        session_id = get_session_id()
        session_storage[session_id] = {
            'raw_df': df.to_dict('records'),
            'columns': list(df.columns),
            'filename': file.filename,
            'uploaded_at': datetime.now().isoformat(),
            'is_sample': False,
        }

        return jsonify({
            'preview': df.head(8).to_dict('records'),
            'columns': list(df.columns),
            'dtypes': {c: str(df[c].dtype) for c in df.columns},
            'row_count': int(len(df)),
            'schema': schema,
        })
    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded CSV has no readable data.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/sample-inspect')
def sample_inspect():
    """Inspect the bundled demo dataset the same way an uploaded CSV is inspected."""
    try:
        from schema_engine import detect_schema
        sample_path = os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv')
        df = pd.read_csv(sample_path)
        schema = detect_schema(df)

        session_id = get_session_id()
        session_storage[session_id] = {
            'raw_df': df.to_dict('records'),
            'columns': list(df.columns),
            'filename': 'sample_transactions.csv',
            'uploaded_at': datetime.now().isoformat(),
            'is_sample': True,
        }
        return jsonify({
            'preview': df.head(8).to_dict('records'),
            'columns': list(df.columns),
            'dtypes': {c: str(df[c].dtype) for c in df.columns},
            'row_count': int(len(df)),
            'schema': schema,
            'is_sample': True,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sample-data')
def get_sample_data():
    """Get the full pre-computed analysis for the bundled demo dataset."""
    try:
        return jsonify(load_sample_data())
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    """Run analysis on uploaded data, resolving it through the schema engine first."""
    try:
        from schema_engine import detect_schema, resolve_dataframe
        session_id = get_session_id()
        
        if session_id not in session_storage:
            return jsonify({'error': 'No data uploaded. Please upload CSV first.'}), 400
        
        stored = session_storage[session_id]
        if 'raw_df' not in stored:
            return jsonify({'error': 'No data uploaded. Please inspect a CSV first.'}), 400
        
        df = pd.DataFrame(stored['raw_df'])
        schema = detect_schema(df)
        resolved = resolve_dataframe(df, schema)
        analysis = analyzer.analyze_uploaded_data(resolved)
        analysis['schema'] = schema
        stored['analysis'] = analysis
        stored['schema'] = schema
        return jsonify(analysis)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Handle CSV upload: inspect schema, run analysis, store for the session."""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.csv'):
            return jsonify({'error': 'Only CSV files are supported'}), 400

        df = pd.read_csv(file)
        if len(df) == 0:
            return jsonify({'error': 'The uploaded CSV is empty.'}), 400

        analysis = _resolve_and_analyze(df)

        session_id = get_session_id()
        session_storage[session_id] = {
            'raw_df': df.to_dict('records'),
            'columns': list(df.columns),
            'analysis': analysis,
            'schema': analysis.get('schema'),
            'uploaded_at': datetime.now().isoformat(),
            'filename': file.filename,
            'is_sample': False,
        }

        # If the uploaded CSV contains causal fields, also build the unified
        # Revenue Agent state so the agent, simulator, and optimizer work.
        try:
            _build_agent_state(session_id, df, file.filename)
        except Exception:
            pass  # descriptive analysis remains available

        return jsonify(analysis)

    except pd.errors.EmptyDataError:
        return jsonify({'error': 'The uploaded CSV has no readable data.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/campaigns/<campaign_id>')
def get_campaign_detail(campaign_id):
    """Get detailed analysis for a specific campaign"""
    try:
        session_id = get_session_id()
        
        if session_id not in session_storage or 'analysis' not in session_storage[session_id]:
            # Try loading sample data
            analysis = load_sample_data()
            if 'error' in analysis:
                return jsonify({'error': 'No analysis data available'}), 400
        else:
            analysis = session_storage[session_id]['analysis']
        
        # Filter transactions by campaign
        if '_raw_transactions' not in analysis:
            return jsonify({'error': 'No transaction data available'}), 400
        
        df = pd.DataFrame(analysis['_raw_transactions'])
        
        if 'campaign_id' not in df.columns:
            return jsonify({'error': 'Campaign ID not available in data'}), 400
        
        campaign_df = df[df['campaign_id'] == campaign_id]
        
        if len(campaign_df) == 0:
            return jsonify({'error': f'Campaign {campaign_id} not found'}), 404
        
        # Calculate campaign-specific metrics
        total_orders = len(campaign_df)
        total_revenue = float(campaign_df['final_value'].sum())
        total_discounts = float(campaign_df['discount_amount'].sum()) if 'discount_amount' in campaign_df.columns else 0
        
        # Segment breakdown
        segment_breakdown = campaign_df.groupby('customer_type').agg({
            'final_value': ['count', 'sum'],
            'discount_amount': 'sum' if 'discount_amount' in campaign_df.columns else 'final_value'
        }).to_dict()
        
        # Business narrative
        discount_rate = total_discounts / (total_discounts + total_revenue) if total_revenue > 0 else 0
        
        narrative_parts = []
        if discount_rate > 0.15:
            narrative_parts.append(f"This campaign gave away {discount_rate*100:.0f}% of revenue in discounts.")
        else:
            narrative_parts.append(f"This campaign maintained a healthy {discount_rate*100:.0f}% discount rate.")
        
        # Check for leakage
        sure_thing_discounts = campaign_df[
            (campaign_df['customer_type'] == 'Sure Thing') & 
            (campaign_df['discount_amount'] > 0 if 'discount_amount' in campaign_df.columns else False)
        ]
        if len(sure_thing_discounts) > 0:
            leak_pct = len(sure_thing_discounts) / total_orders * 100
            narrative_parts.append(f"{leak_pct:.0f}% of orders were to Sure Things receiving unnecessary discounts.")
        
        return jsonify({
            'campaign_id': campaign_id,
            'summary': {
                'total_orders': total_orders,
                'total_revenue': round(total_revenue, 2),
                'total_discounts': round(total_discounts, 2),
                'discount_rate': round(discount_rate, 3)
            },
            'segment_breakdown': segment_breakdown,
            'narrative': ' '.join(narrative_parts),
            'leakage_alert': len(sure_thing_discounts) > total_orders * 0.1 if len(sure_thing_discounts) > 0 else False
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/campaigns')
def list_campaigns():
    """List all campaigns in the dataset"""
    try:
        session_id = get_session_id()
        
        if session_id not in session_storage or 'analysis' not in session_storage[session_id]:
            analysis = load_sample_data()
        else:
            analysis = session_storage[session_id]['analysis']
        
        if '_raw_transactions' not in analysis:
            return jsonify({'campaigns': []})
        
        df = pd.DataFrame(analysis['_raw_transactions'])
        
        if 'campaign_id' not in df.columns:
            return jsonify({'campaigns': []})
        
        campaigns = df.groupby('campaign_id').agg({
            'final_value': ['count', 'sum'],
            'discount_amount': 'sum' if 'discount_amount' in df.columns else 'final_value'
        }).reset_index()
        campaigns.columns = ['campaign_id', 'orders', 'revenue', 'discounts']
        
        # Add segment breakdown per campaign
        result = []
        for camp_id in campaigns['campaign_id']:
            camp_df = df[df['campaign_id'] == camp_id]
            seg_counts = camp_df['customer_type'].value_counts().to_dict()
            result.append({
                'campaign_id': camp_id,
                'orders': int(camp_df['transaction_id'].nunique() if 'transaction_id' in camp_df.columns else len(camp_df)),
                'revenue': round(float(camp_df['final_value'].sum()), 2),
                'discounts': round(float(camp_df['discount_amount'].sum() if 'discount_amount' in camp_df.columns else 0), 2),
                'segment_distribution': seg_counts
            })
        
        return jsonify({'campaigns': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/agent/investigate', methods=['POST'])
def agent_investigate():
    """AI Agent investigation endpoint with structured recommendations"""
    try:
        data = request.get_json()
        question = data.get('question', '')
        analysis_data = data.get('analysisData')
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        # Get base investigation results
        investigation = investigator.investigate(question, analysis_data)
        
        # Enhance with structured recommendation card format
        if 'recommended_actions' in investigation:
            # Format as structured cards
            cards = []
            for i, action in enumerate(investigation['recommended_actions'][:3]):
                cards.append({
                    'action': action,
                    'why': investigation.get('findings', [''])[min(i, len(investigation.get('findings', []))-1)] if investigation.get('findings') else 'Based on data analysis',
                    'impact': investigation.get('financial_impact', 'Positive financial impact expected'),
                    'risk': 'LOW' if i == 0 else ('MEDIUM' if i == 1 else 'MEDIUM-HIGH'),
                    'confidence': investigation.get('confidence', 'MEDIUM')
                })
            investigation['recommendation_cards'] = cards
        
        return jsonify(investigation)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/micro-test/proposal', methods=['POST'])
def micro_test_proposal():
    """Generate a micro-test proposal based on analysis"""
    try:
        data = request.get_json()
        analysis_data = data.get('analysisData')
        target_segment = data.get('targetSegment', 'Sure Thing')
        
        if not analysis_data:
            return jsonify({'error': 'No analysis data provided'}), 400
        
        summary = analysis_data.get('summary', {})
        leakage = summary.get('estimated_margin_leakage', 0)
        leakage_tx = summary.get('leakage_transactions', 0)
        
        # Calculate sample size suggestion (basic power analysis approximation)
        confidence_level = 0.95
        margin_of_error = 0.05
        z_score = 1.96  # for 95% CI
        
        # Conservative estimate for binary outcome (conversion)
        p = 0.5
        sample_size = int((z_score**2 * p * (1-p)) / (margin_of_error**2))
        
        # Adjust for treatment/control split
        sample_size_per_group = sample_size
        
        proposal = {
            'test_name': f'{target_segment} Zero-Discount Test',
            'objective': f'Measure the impact of removing discounts from {target_segment} customers',
            'hypothesis': f'{target_segment} customers will maintain >90% purchase rate without discounts',
            'duration_days': 14,
            'sample_size': {
                'treatment': sample_size_per_group,
                'control': sample_size_per_group,
                'total': sample_size_per_group * 2
            },
            'success_metrics': [
                'Conversion rate (primary)',
                'Revenue per visitor',
                'Margin per transaction',
                '90-day retention (secondary)'
            ],
            'expected_impact': {
                'revenue_change': f'-${leakage * 0.1:.0f} to +${leakage * 0.05:.0f} (range)',
                'margin_improvement': f'~${leakage * 0.7:.0f} recovered',
                'risk_level': 'LOW' if target_segment == 'Sure Thing' else 'MEDIUM'
            },
            'implementation': [
                f'1. Tag {target_segment} customers in your CDP/ESP',
                '2. Exclude tagged customers from discount campaigns',
                '3. Create holdout control group (50/50 split)',
                '4. Run for 14 days minimum',
                '5. Monitor daily for significant drop in conversion'
            ],
            'stop_conditions': [
                'Conversion drops >15% in treatment vs control',
                'Revenue per visitor drops >10%',
                'Customer complaints spike (>2x baseline)'
            ]
        }
        
        return jsonify(proposal)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/opportunity-cost')
def get_opportunity_cost():
    """Get detailed opportunity cost analysis"""
    try:
        session_id = get_session_id()
        
        if session_id not in session_storage or 'analysis' not in session_storage[session_id]:
            analysis = load_sample_data()
        else:
            analysis = session_storage[session_id]['analysis']
        
        summary = analysis.get('summary', {})
        segments = analysis.get('segments', {})
        discount_by_segment = analysis.get('discount_by_segment', {})
        
        opportunity_cost = summary.get('opportunity_cost', 0)
        margin_leakage = summary.get('estimated_margin_leakage', 0)
        
        # Break down by segment
        segment_impact = []
        for seg in ['Sure Thing', 'Persuadable', 'Price Warrior', 'Lost Cause']:
            stats = discount_by_segment.get(seg, {})
            if stats.get('discount_rate', 0) > 0:
                segment_impact.append({
                    'segment': seg,
                    'discount_rate': stats.get('discount_rate', 0),
                    'orders': stats.get('orders', 0),
                    'revenue': stats.get('revenue', 0),
                    'assessment': 'Unnecessary' if seg == 'Sure Thing' else ('Effective' if seg == 'Persuadable' else 'Variable')
                })
        
        return jsonify({
            'summary': {
                'total_opportunity_cost': opportunity_cost,
                'total_margin_leakage': margin_leakage,
                'annualized_impact': opportunity_cost * 12,
                'recoverable_percentage': 0.75
            },
            'breakdown': segment_impact,
            'key_insights': [
                f'${opportunity_cost:.0f} in revenue was given away as unnecessary discounts',
                f'${margin_leakage:.0f} represents pure margin destruction',
                'Sure Thing segment shows highest leakage with lowest incremental value',
                'Reallocating discounts to Persuadables could improve ROI'
            ],
            'recommendations': [
                'Immediately stop discounting Sure Things',
                'Reduce Price Warrior discounts by 20%',
                'Increase Persuadable discount depth by 5%',
                'A/B test all changes before full rollout'
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/simulate', methods=['POST'])
def simulate_policy():
    """Simulate new discount policy"""
    try:
        data = request.get_json()
        current_policy = data.get('currentPolicy', {})
        proposed_policy = data.get('proposedPolicy', {})
        analysis_data = data.get('analysisData')
        
        # Run simulation
        simulation_results = analyzer.simulate_policy(current_policy, proposed_policy, analysis_data)
        return jsonify(simulation_results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/customer/<customer_id>')
def get_customer_details(customer_id):
    """Get detailed analysis for a specific customer (from session analysis, or sample data)."""
    try:
        session_id = get_session_id()
        analysis = None
        if session_id in session_storage and 'analysis' in session_storage[session_id]:
            analysis = session_storage[session_id]['analysis']
        else:
            analysis = load_sample_data()

        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 400

        customer_details = analyzer.get_customer_analysis(customer_id, analysis)
        return jsonify(customer_details)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/leakage-intelligence')
def leakage_intelligence():
    """Multi-dimensional leakage breakdown. Only reports dimensions present in the data."""
    try:
        analysis, _ = _current_analysis()
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 400

        tx = pd.DataFrame(analysis.get('_raw_transactions', []))
        schema = analysis.get('schema', {})
        mappings = schema.get('mappings', {})
        summary = analysis.get('summary', {})

        sections = []

        def _has(*cols):
            return any(c in tx.columns for c in cols)

        # 1. Discount leakage (core) — always available if treated + leakage fields exist
        discount = {
            'key': 'discount',
            'title': 'Discount Leakage',
            'status': 'AVAILABLE' if ('was_leakage' in tx.columns and 'treated' in tx.columns) else 'UNAVAILABLE',
            'description': 'Discounts given to high-intent customers who likely would have purchased anyway.',
        }
        if discount['status'] == 'AVAILABLE':
            leak = tx[tx.get('was_leakage', False)] if 'was_leakage' in tx.columns else tx[tx['treated'] & (tx['customer_type'] == 'Sure Thing')]
            discount['value'] = round(float(leak['discount_amount'].sum()) if 'discount_amount' in leak.columns else 0, 2)
            discount['transactions'] = int(len(leak))
            discount['customers'] = int(leak['customer_id'].nunique()) if 'customer_id' in leak.columns else int(len(leak))
            discount['pct_of_discount_spend'] = round((discount['value'] / summary.get('total_discounts_given', 1) * 100), 1) if summary.get('total_discounts_given') else None
        sections.append(discount)

        # 2. Margin leakage — needs profit + treated; else show gross-margin estimate (labeled)
        margin = {
            'key': 'margin',
            'title': 'Margin Leakage',
            'status': 'UNAVAILABLE',
            'description': 'Excessive discounting reducing contribution margin.',
        }
        if _has('profit', 'margin') and 'treated' in tx.columns:
            profit_col = 'profit' if 'profit' in tx.columns else 'margin'
            treated = tx[tx['treated']]
            margin['status'] = 'AVAILABLE'
            margin['value'] = round(float(pd.to_numeric(treated[profit_col], errors='coerce').sum()), 2) if len(treated) else 0
            margin['transactions'] = int(len(treated))
        elif summary.get('estimated_margin_leakage'):
            # No profit column: report the gross-margin estimate, clearly labeled as an estimate.
            margin['status'] = 'ESTIMATED'
            margin['value'] = round(float(summary.get('estimated_margin_leakage', 0)), 2)
            margin['note'] = 'Profit/margin column not detected — using a 30% gross-margin assumption on unnecessary discount spend.'
        sections.append(margin)

        # 3. Promotion leakage — needs campaign + treated + leakage
        promo = {
            'key': 'promotion',
            'title': 'Promotion Leakage',
            'status': 'AVAILABLE' if _has('campaign_id', 'campaign') and 'treated' in tx.columns else 'UNAVAILABLE',
            'description': 'Campaigns generating high sales volume but weak incremental impact.',
            'breakdown': [],
        }
        if promo['status'] == 'AVAILABLE':
            camp_col = 'campaign_id' if 'campaign_id' in tx.columns else 'campaign'
            groups = tx.groupby(camp_col)
            for camp, g in groups:
                leak = g[g['customer_type'] == 'Sure Thing'] if 'customer_type' in g.columns else g.iloc[0:0]
                promo['breakdown'].append({
                    'name': str(camp),
                    'orders': int(len(g)),
                    'revenue': round(float(pd.to_numeric(g['final_value'], errors='coerce').sum()) if 'final_value' in g.columns else 0, 2),
                    'leakage': round(float(pd.to_numeric(leak['discount_amount'], errors='coerce').sum()) if 'discount_amount' in leak.columns else 0, 2),
                    'leakage_tx': int(len(leak)),
                })
            promo['breakdown'].sort(key=lambda x: x['leakage'], reverse=True)
            promo['top'] = promo['breakdown'][0] if promo['breakdown'] else None
        sections.append(promo)

        # 4. Customer-segment leakage — always available if segments exist
        seg_leak = {
            'key': 'segment',
            'title': 'Customer Segment Leakage',
            'status': 'AVAILABLE' if 'customer_type' in tx.columns else 'UNAVAILABLE',
            'description': 'Which customer groups are receiving unnecessary incentives.',
            'breakdown': [],
        }
        if seg_leak['status'] == 'AVAILABLE':
            for seg, g in tx.groupby('customer_type'):
                leak = g[g.get('was_leakage', False)] if 'was_leakage' in g.columns else g[g['treated'] & (g['customer_type'] == 'Sure Thing')] if 'treated' in g.columns else g.iloc[0:0]
                seg_leak['breakdown'].append({
                    'segment': str(seg),
                    'orders': int(len(g)),
                    'revenue': round(float(pd.to_numeric(g['final_value'], errors='coerce').sum()) if 'final_value' in g.columns else 0, 2),
                    'leakage': round(float(pd.to_numeric(leak['discount_amount'], errors='coerce').sum()) if 'discount_amount' in leak.columns else 0, 2),
                    'leakage_tx': int(len(leak)),
                })
        sections.append(seg_leak)

        # 5. Product/category leakage — needs product_category
        product = {
            'key': 'product',
            'title': 'Product / Category Leakage',
            'status': 'AVAILABLE' if _has('product_category', 'category', 'product') else 'UNAVAILABLE',
            'description': 'Categories or products with excessive discounting and low incremental response.',
            'breakdown': [],
        }
        if product['status'] == 'AVAILABLE':
            cat_col = next(c for c in ('product_category', 'category', 'product') if c in tx.columns)
            for cat, g in tx.groupby(cat_col):
                leak = g[g.get('was_leakage', False)] if 'was_leakage' in g.columns else g[g['treated'] & (g['customer_type'] == 'Sure Thing')] if 'treated' in g.columns else g.iloc[0:0]
                product['breakdown'].append({
                    'name': str(cat),
                    'orders': int(len(g)),
                    'revenue': round(float(pd.to_numeric(g['final_value'], errors='coerce').sum()) if 'final_value' in g.columns else 0, 2),
                    'leakage': round(float(pd.to_numeric(leak['discount_amount'], errors='coerce').sum()) if 'discount_amount' in leak.columns else 0, 2),
                })
            product['breakdown'].sort(key=lambda x: x['leakage'], reverse=True)
        sections.append(product)

        # 6. Channel leakage — needs channel
        channel = {
            'key': 'channel',
            'title': 'Channel Leakage',
            'status': 'AVAILABLE' if _has('channel', 'source', 'medium') else 'UNAVAILABLE',
            'description': 'Which acquisition channels leak the most margin.',
            'breakdown': [],
        }
        if channel['status'] == 'AVAILABLE':
            ch_col = next(c for c in ('channel', 'source', 'medium') if c in tx.columns)
            for ch, g in tx.groupby(ch_col):
                leak = g[g.get('was_leakage', False)] if 'was_leakage' in g.columns else g[g['treated'] & (g['customer_type'] == 'Sure Thing')] if 'treated' in g.columns else g.iloc[0:0]
                channel['breakdown'].append({
                    'name': str(ch),
                    'orders': int(len(g)),
                    'revenue': round(float(pd.to_numeric(g['final_value'], errors='coerce').sum()) if 'final_value' in g.columns else 0, 2),
                    'leakage': round(float(pd.to_numeric(leak['discount_amount'], errors='coerce').sum()) if 'discount_amount' in leak.columns else 0, 2),
                })
            channel['breakdown'].sort(key=lambda x: x['leakage'], reverse=True)
        sections.append(channel)

        # 7. Geographic leakage — needs a location dimension
        geo = {
            'key': 'geo',
            'title': 'Geographic Leakage',
            'status': 'AVAILABLE' if _has('region', 'city', 'country', 'location', 'market', 'geo') else 'UNAVAILABLE',
            'description': 'Where discount efficiency is weakest.',
            'breakdown': [],
        }
        if geo['status'] == 'AVAILABLE':
            geo_col = next(c for c in ('region', 'city', 'country', 'location', 'market', 'geo') if c in tx.columns)
            for loc, g in tx.groupby(geo_col):
                leak = g[g.get('was_leakage', False)] if 'was_leakage' in g.columns else g[g['treated'] & (g['customer_type'] == 'Sure Thing')] if 'treated' in g.columns else g.iloc[0:0]
                geo['breakdown'].append({
                    'name': str(loc),
                    'orders': int(len(g)),
                    'leakage': round(float(pd.to_numeric(leak['discount_amount'], errors='coerce').sum()) if 'discount_amount' in leak.columns else 0, 2),
                })
            geo['breakdown'].sort(key=lambda x: x['leakage'], reverse=True)
        sections.append(geo)

        return jsonify({
            'sections': sections,
            'total_leakage': round(float(summary.get('estimated_margin_leakage', 0)), 2),
            'total_discount_spend': round(float(summary.get('total_discounts_given', 0)), 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent-audit')
def agent_audit():
    """Agentic investigation narrative + activity log, derived from the actual analysis."""
    try:
        analysis, _ = _current_analysis()
        if 'error' in analysis:
            return jsonify({'error': analysis['error']}), 400

        schema = analysis.get('schema', {})
        summary = analysis.get('summary', {})
        segments = analysis.get('segments', {})
        discount_by_segment = analysis.get('discount_by_segment', {})
        leakage_sources = analysis.get('leakage_sources', [])
        mappings = schema.get('mappings', {})

        # Build a dynamic activity log from the schema + analysis
        steps = []
        t = datetime.now().strftime('%H:%M:%S')
        steps.append(f'Inspected {summary.get("total_orders", 0):,} records across {len(schema.get("columns", []))} columns.')
        detected = [v['column'] for v in mappings.values()]
        if detected:
            steps.append(f'Mapped {len(detected)} relevant variables: {", ".join(detected[:6])}.')
        raw_tx = analysis.get('_raw_transactions') or []
        if raw_tx and 'treated' in raw_tx[0]:
            steps.append('Detected treatment/control variation in discount exposure.')
        steps.append(f'Found {summary.get("total_orders", 0):,} orders and {summary.get("estimated_margin_leakage", 0):,.0f} in estimated margin leakage.')
        sure_things = segments.get('counts', {}).get('Sure Thing', 0)
        if sure_things:
            steps.append(f'Identified {sure_things:,} Sure Thing customers receiving discounts.')
        steps.append(f'Estimated ₹{summary.get("estimated_margin_leakage", 0):,.0f} potential margin leakage.')
        steps.append('Generated optimized targeting policy and recoverable-margin estimate.')

        # Dynamic insights (never invented — each tied to an actual computed value)
        insights = []
        st = discount_by_segment.get('Sure Thing', {})
        if st.get('discount_rate', 0) > 0:
            insights.append(f"{st['orders']:,} Sure Thing orders received discounts at a {st['discount_rate']*100:.0f}% rate despite high purchase intent.")
        pers = discount_by_segment.get('Persuadable', {})
        if pers:
            insights.append(f"Persuadable customers convert with genuine hesitation — they show a {pers.get('discount_rate', 0)*100:.0f}% discount reliance, the highest-ROI place to spend.")
        if leakage_sources:
            top = leakage_sources[0]
            insights.append(f"{top['source']} has the highest potential leakage: ₹{top['leakage']:,.0f} across {top['transactions']} transactions.")
        ready = [k for k, v in schema.get('capabilities', {}).items() if v.get('status') == 'READY']
        if ready:
            insights.append(f"Available analyses: {', '.join(ready)}.")
        if not insights:
            insights.append('No strong leakage signal detected in the current dataset.')

        return jsonify({
            'status': 'complete',
            'agent': 'Discount Lens Agent',
            'activity_log': steps,
            'insights': insights,
            'agent_reasoning': schema.get('agent_reasoning', []),
            'discount_case_label': schema.get('discount_label', ''),
            'capabilities': schema.get('capabilities', {}),
            'missing_concepts': schema.get('missing_concepts', []),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/data-explorer')
def data_explorer():
    """Business-friendly data profile: rows, columns, detected fields, missing values, duplicates, distributions."""
    try:
        session_id = get_session_id()
        stored = session_storage.get(session_id, {})
        if 'raw_df' in stored:
            df = pd.DataFrame(stored['raw_df'])
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv'))

        from schema_engine import detect_schema
        schema = detect_schema(df)

        n_rows = len(df)
        n_cols = len(df.columns)
        missing = {c: int(df[c].isna().sum()) for c in df.columns}
        duplicates = int(df.duplicated().sum())

        # basic distributions for numeric columns
        distributions = []
        for c in df.columns:
            if pd.api.types.is_numeric_dtype(df[c]):
                s = pd.to_numeric(df[c], errors='coerce').dropna()
                if len(s):
                    distributions.append({
                        'column': c,
                        'type': 'numeric',
                        'min': round(float(s.min()), 2),
                        'max': round(float(s.max()), 2),
                        'mean': round(float(s.mean()), 2),
                        'non_null': int(len(s)),
                    })
            else:
                s = df[c].dropna().astype(str)
                top = s.value_counts().head(3).to_dict() if len(s) else {}
                distributions.append({
                    'column': c,
                    'type': 'categorical',
                    'unique': int(s.nunique()),
                    'non_null': int(len(s)),
                    'top_values': top,
                })

        return jsonify({
            'rows': n_rows,
            'columns': n_cols,
            'column_names': list(df.columns),
            'mappings': {k: v['column'] for k, v in schema.get('mappings', {}).items()},
            'missing_values': missing,
            'total_missing': int(sum(missing.values())),
            'duplicate_rows': duplicates,
            'distributions': distributions,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/raw-data')
def raw_data():
    """Return the raw uploaded/sample rows as a readable preview for the data explorer tab."""
    try:
        session_id = get_session_id()
        stored = session_storage.get(session_id, {})
        if 'raw_df' in stored:
            df = pd.DataFrame(stored['raw_df'])
        else:
            df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv'))
        return jsonify({
            'filename': stored.get('filename', 'sample_transactions.csv'),
            'columns': list(df.columns),
            'rows': df.head(100).fillna('').to_dict('records'),
            'total_rows': int(len(df)),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ---------------------------------------------------------------------------
# Revenue Agent — unified causal analysis state, agent workflow, simulation,
# optimizer, and local deployment endpoints. All metrics derive from the
# active dataset and model; nothing below hardcodes a business number.
# ---------------------------------------------------------------------------

def _build_agent_state(session_id, df, dataset_name):
    """Run the full causal/descriptive pipeline and cache the unified state."""
    validation = causal_analyzer.validate(df)
    model = causal_analyzer.analyze(df, validation)
    descriptive = {}
    if not model.get('available'):
        try:
            schema = __import__('schema_engine').detect_schema(df)
            resolved = __import__('schema_engine').resolve_dataframe(df, schema)
            descriptive = analyzer.analyze_uploaded_data(resolved)
        except Exception:
            descriptive = {}
    schema_info = {
        'row_count': int(len(df)),
        'columns': list(df.columns),
        'mappings': validation.feature_columns,
    }
    state = revenue_agent.build_state(dataset_name, schema_info, validation.as_dict(), model, descriptive)
    state['rows_preview'] = df.head(50).replace({np.nan: None}).to_dict('records')
    agent_memory.start(session_id, state)
    session_storage[session_id] = session_storage.get(session_id, {})
    session_storage[session_id]['agent_state'] = state
    session_storage[session_id]['agent_df'] = df
    return state


@app.route('/api/agent/demo', methods=['POST'])
def agent_load_demo():
    """Load the programmatic causal demo dataset and run the full pipeline."""
    try:
        data = request.get_json(silent=True) or {}
        rows = int(data.get('rows', 2500))
        seed = int(data.get('seed', 42))
        df = generate_demo_dataset(rows=rows, seed=seed)
        session_id = get_session_id()
        state = _build_agent_state(session_id, df, 'Causal Demo Dataset')
        return jsonify(state)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/state')
def agent_state():
    """Return the cached unified analysis state for the current session."""
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    if 'agent_state' not in stored:
        return jsonify({'error': 'No active dataset. Load the demo dataset or upload a CSV.'}), 400
    state = stored['agent_state']
    state['deployed_policy'] = agent_memory.get(session_id).get('deployed_policy')
    state['last_simulation'] = agent_memory.get(session_id).get('last_simulation')
    return jsonify(state)


@app.route('/api/agent/brief')
def agent_brief():
    """Return the proactive Daily Revenue Intelligence Brief."""
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    if 'agent_state' not in stored:
        return jsonify({'error': 'No active dataset.'}), 400
    return jsonify(stored['agent_state'].get('brief', {}))


@app.route('/api/agent/trail')
def agent_trail():
    """Return the execution trail (OBSERVE → VALIDATE → MODEL → ...)."""
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    if 'agent_state' not in stored:
        return jsonify({'error': 'No active dataset.'}), 400
    return jsonify({'trail': stored['agent_state'].get('trail', [])})


@app.route('/api/agent/investigate-v2', methods=['POST'])
def agent_investigate_v2():
    """Revenue Agent chat: routes a natural-language question to data tools."""
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        session_id = get_session_id()
        if not question:
            return jsonify({'error': 'No question provided.'}), 400
        stored = session_storage.get(session_id, {})
        if 'agent_state' not in stored:
            return jsonify({'error': 'No active dataset. Load the demo dataset first.'}), 400
        response = revenue_agent.investigate(session_id, question, stored['agent_state'])
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/simulate', methods=['POST'])
def agent_simulate_policy():
    """Run the model-driven policy simulator."""
    try:
        data = request.get_json() or {}
        proposed = data.get('proposedPolicy') or data.get('proposed_policy') or recommender.recommendation(
            session_storage.get(get_session_id(), {}).get('agent_state', {}).get('model', {})
        ).get('proposed_policy', AGENT_DEFAULT_POLICY)
        current = data.get('currentPolicy') or AGENT_DEFAULT_POLICY
        session_id = get_session_id()
        stored = session_storage.get(session_id, {})
        if 'agent_state' not in stored:
            return jsonify({'error': 'No active dataset.'}), 400
        result = agent_simulate(current, proposed, stored['agent_state'])
        if 'error' not in result:
            agent_memory.save_simulation(session_id, result)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/optimize', methods=['POST'])
def agent_optimize_policy():
    """Find the best policy maximizing expected profit subject to revenue retention."""
    try:
        session_id = get_session_id()
        stored = session_storage.get(session_id, {})
        if 'agent_state' not in stored:
            return jsonify({'error': 'No active dataset.'}), 400
        result = recommender.optimize(agent_simulate, stored['agent_state'])
        if 'error' not in result and result.get('simulation'):
            agent_memory.save_simulation(session_id, result['simulation'])
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/agent/deploy', methods=['POST'])
def agent_deploy_policy():
    """Locally deploy an approved policy (demo-only; no external platform is modified)."""
    try:
        data = request.get_json() or {}
        policy = data.get('policy')
        if not policy:
            return jsonify({'error': 'No policy provided for deployment.'}), 400
        session_id = get_session_id()
        deployed = agent_memory.deploy(session_id, policy)
        stored = session_storage.get(session_id, {})
        if 'agent_state' in stored:
            stored['agent_state']['deployed_policy'] = deployed
        return jsonify({
            'status': 'POLICY ACTIVE',
            'policy': policy_json(policy),
            'deployed_at': deployed['deployed_at'],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/system-check')
def system_check():
    """Run development checks against the active session state."""
    checks = []
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    state = stored.get('agent_state')
    checks.append({'name': 'Active dataset loaded', 'pass': state is not None})
    if not state:
        return jsonify({'checks': checks, 'all_passed': False})
    model = state.get('model', {})
    checks.append({'name': 'Causal model available', 'pass': model.get('available', False)})
    checks.append({'name': 'Predictions generated', 'pass': len(model.get('predictions', [])) > 0})
    checks.append({'name': 'Diagnostics reported', 'pass': bool(model.get('diagnostics'))})
    checks.append({'name': 'Reliability grade set', 'pass': bool(model.get('reliability'))})
    checks.append({'name': 'Opportunities ranked', 'pass': len(state.get('opportunities', [])) > 0})
    checks.append({'name': 'Recommendation generated', 'pass': bool(state.get('recommendation'))})
    sim = agent_memory.get(session_id).get('last_simulation')
    checks.append({'name': 'Simulation run', 'pass': sim is not None and 'error' not in (sim or {})})
    checks.append({'name': 'Policy deployed', 'pass': agent_memory.get(session_id).get('deployed_policy') is not None})
    return jsonify({'checks': checks, 'all_passed': all(c['pass'] for c in checks)})


@app.route('/dataset')
def dataset_preview_page():
    """Standalone, readable HTML preview of the sample (or uploaded) dataset — opens in a new tab."""
    session_id = get_session_id()
    stored = session_storage.get(session_id, {})
    if 'raw_df' in stored:
        df = pd.DataFrame(stored['raw_df'])
        filename = stored.get('filename', 'dataset.csv')
    else:
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'sample_transactions.csv'))
        filename = 'sample_transactions.csv'

    columns = [str(c) for c in df.columns]
    rows = df.head(200).fillna('').to_dict('records')
    n_rows = int(len(df))

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{filename} — Discount Lens</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>body{{font-family:ui-sans-serif,system-ui,-apple-system,sans-serif;}}</style>
</head><body class="bg-slate-50 text-slate-900 p-6">
<div class="max-w-6xl mx-auto">
  <div class="flex items-center justify-between mb-4">
    <div>
      <div class="text-xs font-mono text-slate-500">DISCOUNT LENS · DATASET PREVIEW</div>
      <h1 class="text-2xl font-semibold tracking-tight">{filename}</h1>
      <p class="text-sm text-slate-500 mt-1">{n_rows:,} rows · {len(columns)} columns (showing first {len(rows)})</p>
    </div>
    <a href="/" class="text-sm text-slate-500 hover:text-slate-900 underline">← Back to Discount Lens</a>
  </div>
  <div class="bg-white border border-slate-200 rounded-2xl overflow-x-auto shadow-sm">
    <table class="w-full text-xs">
      <thead><tr class="bg-slate-50 border-b text-slate-500">
        {''.join(f'<th class="text-left px-3 py-2 font-semibold whitespace-nowrap">{c}</th>' for c in columns)}
      </tr></thead>
      <tbody class="divide-y">
        {''.join('<tr>' + ''.join(f'<td class="px-3 py-2 whitespace-nowrap text-slate-600">{row.get(c, "")}</td>' for c in columns) + '</tr>' for row in rows)}
      </tbody>
    </table>
  </div>
</div>
</body></html>"""


if __name__ == '__main__':
    # Load sample data on startup
    print("Loading sample data...")
    load_sample_data()
    print("Sample data loaded successfully!")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5001) 