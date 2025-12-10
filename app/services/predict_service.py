# app/services/predict_service.py
import os
import joblib
import pandas as pd

MODEL_PATH = "app/ml/product_model.pkl"
SCALER_PATH = "app/ml/scaler.pkl"

FEATURES = [
    "searched", "in_favorite", "unit_price",
    "original_price", "discount_rate", "rating_average", "review_count",
    "is_top_brand", "is_freeship_xtra", "quantity_sold", "has_buynow",
    "day_ago_created", "qty_clicks"
]


def _prepare_input(data: dict):
    # fill missing, coerce to numeric
    row = {}
    for f in FEATURES:
        v = data.get(f, 0)
        # if string empty -> 0
        if v in ["", None]:
            v = 0
        try:
            row[f] = float(v)
        except Exception:
            # some booleans may be strings like '0'/'1'
            try:
                row[f] = float(str(v).strip())
            except Exception:
                row[f] = 0.0
    return pd.DataFrame([row])


def predict_product(data: dict):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return {"error": "Model chưa được train (product_model.pkl hoặc scaler.pkl không tồn tại)"}

    df = _prepare_input(data)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    Xs = scaler.transform(df[FEATURES])
    prob = model.predict_proba(Xs)[0][1]
    return {"probability": round(float(prob), 6)}
