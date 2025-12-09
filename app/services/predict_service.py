import pandas as pd
import joblib
import os

MODEL_PATH = "app/ml/product_model.pkl"
SCALER_PATH = "app/ml/scaler.pkl"

required_features = [
    "in_favorite", "discount_rate", "rating_average", "unit_price",
    "seller_id", "price", "original_price", "discount", "review_count",
    "is_top_brand", "quantity_sold", "qty_clicks"
]


def predict_product(data: dict):
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model chưa được train"}

    # Thiếu giá trị → tự điền
    for col in required_features:
        if col not in data or data[col] in ["", None]:
            data[col] = 0

    df = pd.DataFrame([data])
    df = df[required_features].fillna(0)

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    prob = model.predict_proba(scaler.transform(df))[0][1]

    return {"probability": round(float(prob), 6)}
