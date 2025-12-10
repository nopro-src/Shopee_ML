# app/services/train_service.py
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

MODEL_PATH = "app/ml/product_model.pkl"
SCALER_PATH = "app/ml/scaler.pkl"

# Final feature set chosen (exists in your CSV)
FEATURES = [
    "searched",        # user searched product
    "in_favorite",     # added to wishlist/favorite
    "unit_price",      # price per unit
    "original_price",  # gốc để tính khuyến mại
    "discount_rate",   # phần trăm giảm giá
    "rating_average",
    "review_count",
    "is_top_brand",
    "is_freeship_xtra",
    "quantity_sold",
    "has_buynow",
    "day_ago_created",
    "qty_clicks"
]


def _ensure_columns(df: pd.DataFrame):
    # Thêm cột thiếu với giá trị 0
    for c in FEATURES + ["purchased"]:
        if c not in df.columns:
            df[c] = 0
    return df


def _to_numeric(df: pd.DataFrame):
    # Chuyển các cột cần thiết về numeric, giữ 0 nếu lỗi
    for c in FEATURES:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    # target
    df["purchased"] = pd.to_numeric(df["purchased"], errors="coerce").fillna(0).astype(int)
    return df


def train_product_model(csv_path: str):
    """
    Train model và lưu model + scaler. Trả về dict metrics để hiển thị.
    """
    # Load
    df = pd.read_csv(csv_path, low_memory=False)

    # Ensure expected columns exist
    df = _ensure_columns(df)

    # Convert to numeric / fillna
    df = _to_numeric(df)

    # X, y
    X = df[FEATURES]
    y = df["purchased"]

    # Train/test split (stratify nếu có đủ class)
    stratify_param = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE (chỉ khi imbalance > 1)
    try:
        smote = SMOTE(random_state=42)
        X_r, y_r = smote.fit_resample(X_train_scaled, y_train)
    except Exception:
        # Nếu SMOTE fail (ví dụ lớp ít sample), dùng X_train_scaled
        X_r, y_r = X_train_scaled, y_train

    # Train model
    model = LogisticRegression(max_iter=5000)
    model.fit(X_r, y_r)

    # Predict & metrics
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    roc = None
    try:
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        roc = None

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # Save artifacts
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    # return metrics
    return {
        "roc_auc": roc,
        "report": report,
        "n_rows": len(df),
        "features": FEATURES
    }
