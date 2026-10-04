"""Delay-risk model: predicts the probability that an order will be delivered late."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

CAT = ["warehouse_name", "carrier_name", "product_category", "destination_zone", "day_of_week", "season"]
NUM = ["distance_km", "weight_kg", "order_value", "processing_days"]


def _prep(df: pd.DataFrame):
    X = df[CAT + NUM].copy()
    X["day_of_week"] = X["day_of_week"].astype(str)
    return X, df["is_late"]


def _pipe(model, scale=False):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
            ("num", StandardScaler() if scale else "passthrough", NUM),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def train_models(df: pd.DataFrame):
    """Train 3 models, return fitted best model + comparison table + tree rules."""
    X, y = _prep(df)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    candidates = {
        "Logistic Regression": _pipe(LogisticRegression(max_iter=1000), scale=True),
        "Decision Tree (depth 4)": _pipe(DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=42)),
        "Gradient Boosting": _pipe(GradientBoostingClassifier(random_state=42)),
    }
    rows, fitted = [], {}
    for name, pipe in candidates.items():
        pipe.fit(Xtr, ytr)
        proba = pipe.predict_proba(Xte)[:, 1]
        rows.append(
            dict(
                model=name,
                accuracy=round(accuracy_score(yte, proba > 0.5), 3),
                roc_auc=round(roc_auc_score(yte, proba), 3),
            )
        )
        fitted[name] = pipe
    results = pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = results.loc[0, "model"]

    # feature importance from logistic regression coefficients (interpretable)
    lr = fitted["Logistic Regression"]
    names = lr.named_steps["pre"].get_feature_names_out()
    coefs = pd.DataFrame({"feature": names, "coef": lr.named_steps["model"].coef_[0]})
    coefs["feature"] = coefs["feature"].str.replace("cat__", "", regex=False).str.replace("num__", "", regex=False)
    coefs["abs"] = coefs["coef"].abs()
    top_drivers = coefs.sort_values("abs", ascending=False).head(12).drop(columns="abs").round(3)

    # readable decision-tree rules
    tree = fitted["Decision Tree (depth 4)"]
    rules = export_text(
        tree.named_steps["model"], feature_names=list(tree.named_steps["pre"].get_feature_names_out()), max_depth=3
    )
    return fitted[best_name], best_name, results, top_drivers, rules


def predict_risk(model, row: dict) -> float:
    X = pd.DataFrame([row])
    return float(model.predict_proba(X)[:, 1][0])
