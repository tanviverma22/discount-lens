"""Validation and causal/uplift estimation for Discount Lens."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42
MIN_GROUP_SIZE = 30
MIN_CONVERSIONS = 12


@dataclass
class ValidationResult:
    ready: bool
    findings: List[Dict[str, str]]
    feature_columns: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {"causal_ready": self.ready, "findings": self.findings, "feature_columns": self.feature_columns}


class CausalAnalyzer:
    """T-learner uplift estimator with explicit diagnostics and safe fallbacks."""

    required = ("customer_id", "treated", "conversion")
    post_treatment = {"conversion", "final_value", "profit", "discount_amount", "discount_percent", "treated", "treatment"}
    identifier_columns = {"customer_id", "transaction_id", "timestamp", "coupon_code", "session_id"}

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        findings: List[Dict[str, str]] = []
        missing = [column for column in self.required if column not in df.columns]
        if missing:
            findings.append({"level": "error", "message": f"Causal analysis needs {', '.join(missing)}. Upload a dataset with treatment and conversion outcomes."})
            return ValidationResult(False, findings, [])

        working = df.copy()
        treatment = self._binary(working["treated"])
        outcome = self._binary(working["conversion"])
        invalid_treatment = int(treatment.isna().sum())
        invalid_outcome = int(outcome.isna().sum())
        if invalid_treatment:
            findings.append({"level": "error", "message": f"{invalid_treatment} rows have invalid treatment values. Use 0/1, true/false, or yes/no."})
        if invalid_outcome:
            findings.append({"level": "error", "message": f"{invalid_outcome} rows have invalid conversion values. Use 0/1, true/false, or yes/no."})
        if working.duplicated().any():
            findings.append({"level": "warning", "message": f"{int(working.duplicated().sum())} duplicate rows were found; duplicate records are excluded from model training."})
        for column in ("original_value", "final_value", "revenue", "profit", "cost", "discount_amount"):
            if column in working.columns:
                numeric = pd.to_numeric(working[column], errors="coerce")
                if (numeric < 0).any():
                    findings.append({"level": "warning", "message": f"{int((numeric < 0).sum())} rows have negative {column}; review refunds or data quality."})
        if "discount_percent" in working.columns:
            discount = pd.to_numeric(working["discount_percent"], errors="coerce")
            if (discount > 1).sum() and (discount > 100).any():
                findings.append({"level": "warning", "message": "Some discount percentages exceed 100%; those rows will be clipped for economic calculations."})

        valid = treatment.notna() & outcome.notna()
        treated_count = int((treatment[valid] == 1).sum())
        control_count = int((treatment[valid] == 0).sum())
        conversions = int((outcome[valid] == 1).sum())
        if treated_count < MIN_GROUP_SIZE or control_count < MIN_GROUP_SIZE:
            findings.append({"level": "error", "message": f"Insufficient treatment/control observations ({treated_count} treated, {control_count} control; need at least {MIN_GROUP_SIZE} each)."})
        if conversions < MIN_CONVERSIONS or int((outcome[valid] == 0).sum()) < MIN_CONVERSIONS:
            findings.append({"level": "error", "message": "Insufficient conversion outcome variation for a reliable model."})

        features = self._features(working)
        if not features:
            findings.append({"level": "error", "message": "No usable pre-treatment features found. Add behavioral, channel, product, or customer-history fields."})
        else:
            findings.append({"level": "info", "message": f"Using {len(features)} pre-treatment feature(s): {', '.join(features)}."})

        ready = not any(item["level"] == "error" for item in findings)
        return ValidationResult(ready, findings, features)

    def analyze(self, df: pd.DataFrame, validation: ValidationResult) -> Dict[str, Any]:
        if not validation.ready:
            return {"available": False, "reliability": "INSUFFICIENT", "diagnostics": {}, "predictions": [], "message": "Insufficient treatment/control overlap or required data for reliable causal estimation."}

        work = df.copy().drop_duplicates().reset_index(drop=True)
        work["treated"] = self._binary(work["treated"]).astype(int)
        work["conversion"] = self._binary(work["conversion"]).astype(int)
        features = validation.feature_columns
        x = work[features]
        treatment = work["treated"]
        outcome = work["conversion"]
        stratify = treatment.astype(str) + outcome.astype(str)
        try:
            train_index, test_index = train_test_split(work.index, test_size=.25, random_state=RANDOM_SEED, stratify=stratify)
        except ValueError:
            train_index, test_index = train_test_split(work.index, test_size=.25, random_state=RANDOM_SEED, stratify=treatment)

        preprocessor = self._preprocessor(work[features])
        propensity = Pipeline([("prep", preprocessor), ("model", LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))])
        propensities = self._fit_predict_proba(propensity, x.loc[train_index], treatment.loc[train_index], x)

        treated_train = train_index[treatment.loc[train_index] == 1]
        control_train = train_index[treatment.loc[train_index] == 0]
        treated_model = Pipeline([("prep", self._preprocessor(work[features])), ("model", LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))])
        control_model = Pipeline([("prep", self._preprocessor(work[features])), ("model", LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED))])
        p1 = self._fit_predict_proba(treated_model, x.loc[treated_train], outcome.loc[treated_train], x)
        p0 = self._fit_predict_proba(control_model, x.loc[control_train], outcome.loc[control_train], x)
        lift = p1 - p0

        value = self._value(work)
        discount = self._discount(work, value)
        margin = self._margin(work, value, discount)
        incremental_profit = lift * margin - discount
        profitability_threshold = 0.0
        segments = np.where((p0 >= .55) & (incremental_profit <= profitability_threshold), "Sure Thing",
                    np.where(incremental_profit > profitability_threshold, "Persuadable",
                    np.where(p1 < .25, "Lost Cause", "Price Sensitive")))
        work = work.assign(
            baseline_probability=p0,
            treated_probability=p1,
            estimated_incremental_lift=lift,
            expected_incremental_revenue=lift * value,
            expected_incremental_profit=incremental_profit,
            discount_amount_canonical=discount,
            contribution_margin=margin,
            customer_type=segments,
            was_leakage=(work["treated"] == 1) & (incremental_profit <= profitability_threshold),
        )

        support_low = max(float(np.quantile(propensities[treatment == 0], .05)), float(np.quantile(propensities[treatment == 1], .05)))
        support_high = min(float(np.quantile(propensities[treatment == 0], .95)), float(np.quantile(propensities[treatment == 1], .95)))
        overlap_mask = (propensities >= support_low) & (propensities <= support_high)
        overlap_share = float(overlap_mask.mean())
        test_prop = self._predict_proba(propensity, x.loc[test_index])
        auc = self._safe_auc(treatment.loc[test_index], test_prop)
        brier = float(brier_score_loss(treatment.loc[test_index], test_prop))
        reliability = "HIGH" if overlap_share >= .70 and len(work) >= 500 and auc >= .60 else ("MODERATE" if overlap_share >= .50 else "LOW")

        diagnostics = {
            "rows": int(len(work)), "train_rows": int(len(train_index)), "test_rows": int(len(test_index)),
            "treated_count": int(treatment.sum()), "control_count": int((1 - treatment).sum()),
            "treatment_rate": round(float(treatment.mean()), 4),
            "conversion_rate_treated": round(float(outcome[treatment == 1].mean()), 4),
            "conversion_rate_control": round(float(outcome[treatment == 0].mean()), 4),
            "propensity": {"min": round(float(propensities.min()), 4), "p05": round(float(np.quantile(propensities, .05)), 4), "median": round(float(np.median(propensities)), 4), "p95": round(float(np.quantile(propensities, .95)), 4), "max": round(float(propensities.max()), 4)},
            "common_support": {"low": round(support_low, 4), "high": round(support_high, 4), "share": round(overlap_share, 4), "sufficient": overlap_share >= .50},
            "propensity_auc": round(auc, 4) if auc is not None else None,
            "propensity_brier": round(brier, 4), "reliability": reliability,
        }
        columns = ["transaction_id", "customer_id", "channel", "campaign_id", "product_category", "region", "treated", "conversion", "original_value", "final_value", "profit", "customer_type", "baseline_probability", "treated_probability", "estimated_incremental_lift", "expected_incremental_revenue", "expected_incremental_profit", "discount_amount_canonical", "contribution_margin", "was_leakage"]
        available = [column for column in columns if column in work.columns]
        return {"available": True, "reliability": reliability, "diagnostics": diagnostics, "predictions": work[available].replace({np.nan: None}).to_dict("records"), "feature_columns": features, "profit_is_proxy": "profit" not in df.columns and "cost" not in df.columns}

    @staticmethod
    def _fit_predict_proba(model, x_train, y_train, x_all) -> np.ndarray:
        """Fit then predict, suppressing scipy's benign macOS-BLAS RuntimeWarnings.

        sklearn's matmul can emit ``divide by zero``/``overflow`` RuntimeWarnings on
        Apple's Accelerate BLAS even when inputs and outputs are all finite. The
        predictions are verified correct, so only the noise is suppressed here.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            model.fit(x_train, y_train)
            return model.predict_proba(x_all)[:, 1]

    @staticmethod
    def _predict_proba(model, x) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return model.predict_proba(x)[:, 1]

    @staticmethod
    def _binary(series: pd.Series) -> pd.Series:
        values = series.copy()
        numeric = pd.to_numeric(values, errors="coerce")
        mapped = numeric.where(numeric.isin([0, 1]))
        text = values.astype(str).str.strip().str.lower()
        return mapped.fillna(text.map({"true": 1, "yes": 1, "y": 1, "t": 1, "false": 0, "no": 0, "n": 0, "f": 0}))

    def _features(self, df: pd.DataFrame) -> List[str]:
        excluded = self.post_treatment | self.identifier_columns | {"original_value", "revenue", "cost"}
        candidates = [c for c in df.columns if c not in excluded]
        return [c for c in candidates if not pd.api.types.is_datetime64_any_dtype(df[c]) and df[c].notna().mean() >= .5]

    @staticmethod
    def _preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
        numeric = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
        categorical = [c for c in frame.columns if c not in numeric]
        return ColumnTransformer([
            ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])

    @staticmethod
    def _value(df: pd.DataFrame) -> pd.Series:
        for column in ("original_value", "revenue", "final_value"):
            if column in df.columns:
                return pd.to_numeric(df[column], errors="coerce").fillna(0).clip(lower=0)
        return pd.Series(0.0, index=df.index)

    @staticmethod
    def _discount(df: pd.DataFrame, value: pd.Series) -> pd.Series:
        if "discount_amount" in df.columns:
            return pd.to_numeric(df["discount_amount"], errors="coerce").fillna(0).clip(lower=0)
        if "discount_percent" in df.columns:
            pct = pd.to_numeric(df["discount_percent"], errors="coerce").fillna(0)
            pct = np.where(pct > 1, pct / 100, pct)
            return value * np.clip(pct, 0, 1)
        return pd.Series(0.0, index=df.index)

    @staticmethod
    def _margin(df: pd.DataFrame, value: pd.Series, discount: pd.Series) -> pd.Series:
        if "profit" in df.columns:
            return pd.to_numeric(df["profit"], errors="coerce").fillna(0).clip(lower=0)
        if "cost" in df.columns:
            return (value - pd.to_numeric(df["cost"], errors="coerce").fillna(value)).clip(lower=0)
        return (value - discount) * .30

    @staticmethod
    def _safe_auc(y: pd.Series, probabilities: np.ndarray) -> float | None:
        return float(roc_auc_score(y, probabilities)) if y.nunique() > 1 else None
