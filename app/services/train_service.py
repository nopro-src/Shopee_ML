# app/services/train_service.py
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MODEL_DIR = "app/ml"
os.makedirs(MODEL_DIR, exist_ok=True)

# Product artifacts
PRODUCT_MODEL_PATH = os.path.join(MODEL_DIR, "product_model.pkl")
PRODUCT_SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

# Customer artifacts
CUSTOMER_MODEL_PATH = os.path.join(MODEL_DIR, "customer_model.pkl")        # kmeans
CUSTOMER_SCALER_PATH = os.path.join(MODEL_DIR, "customer_scaler.pkl")      # scaler for aggregation features
CUSTOMER_PCA_PATH = os.path.join(MODEL_DIR, "customer_pca.pkl")            # pca
CUSTOMER_CLUSTER_DF_PATH = os.path.join(MODEL_DIR, "customer_clusters.parquet")

# Feature lists
PRODUCT_FEATURES = [
    "searched", "in_favorite", "unit_price",
    "original_price", "discount_rate", "rating_average", "review_count",
    "is_top_brand", "is_freeship_xtra", "quantity_sold", "has_buynow",
    "day_ago_created", "qty_clicks"
]

CUSTOMER_AGG_FEATURES = ['searched', 'in_cart', 'purchased', 'qty', 'total_price']


def _ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0
    return df


def _to_numeric(df: pd.DataFrame, cols, target=None):
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    if target:
        df[target] = pd.to_numeric(df[target], errors="coerce").fillna(0).astype(int)
    return df


# ---------------- Product training (unchanged largely) ----------------
def train_product_model(csv_path: str):
    df = pd.read_csv(csv_path, low_memory=False)

    df = _ensure_cols(df, PRODUCT_FEATURES + ["purchased"])
    df = _to_numeric(df, PRODUCT_FEATURES, target="purchased")

    X = df[PRODUCT_FEATURES]
    y = df["purchased"]

    stratify_param = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    try:
        smote = SMOTE(random_state=42)
        X_r, y_r = smote.fit_resample(X_train_scaled, y_train)
    except Exception:
        X_r, y_r = X_train_scaled, y_train

    model = LogisticRegression(max_iter=5000, class_weight='balanced', C=0.5)
    model.fit(X_r, y_r)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    joblib.dump(model, PRODUCT_MODEL_PATH)
    joblib.dump(scaler, PRODUCT_SCALER_PATH)

    return {"roc_auc": roc, "report": report, "n_rows": len(df), "features": PRODUCT_FEATURES}


# ---------------- Customer clustering with PCA ----------------
def train_customer_model(csv_path: str, n_clusters: int = 5):
    df = pd.read_csv(csv_path, low_memory=False)

    for c in ['user_id'] + CUSTOMER_AGG_FEATURES:
        if c not in df.columns:
            df[c] = 0

    user_behavior = df.groupby('user_id', dropna=False).agg({
        'searched': 'sum',
        'in_cart': 'sum',
        'purchased': 'sum',
        'qty': 'sum',
        'total_price': 'sum'
    }).reset_index()

    user_behavior[CUSTOMER_AGG_FEATURES] = user_behavior[CUSTOMER_AGG_FEATURES].apply(
        lambda col: pd.to_numeric(col, errors='coerce').fillna(0)
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(user_behavior[CUSTOMER_AGG_FEATURES])

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_pca)
    user_behavior['cluster'] = labels

    # Save artifacts
    joblib.dump(kmeans, CUSTOMER_MODEL_PATH)
    joblib.dump(scaler, CUSTOMER_SCALER_PATH)
    joblib.dump(pca, CUSTOMER_PCA_PATH)
    user_behavior.to_parquet(CUSTOMER_CLUSTER_DF_PATH, index=False)

    # Trả về summary tổng thể cho template
    cluster_summary = user_behavior[CUSTOMER_AGG_FEATURES].mean().round(2).to_dict()

    return {"n_rows": len(user_behavior), "n_clusters": n_clusters, "cluster_summary": cluster_summary}
