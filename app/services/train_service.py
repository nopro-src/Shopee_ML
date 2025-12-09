import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib

MODEL_PATH = "app/ml/product_model.pkl"
SCALER_PATH = "app/ml/scaler.pkl"

FEATURES = [
    'in_favorite', 'discount_rate', 'rating_average', 'unit_price',
    'seller_id', 'price', 'original_price', 'discount', 'review_count',
    'is_top_brand', 'quantity_sold', 'qty_clicks'
]


def train_product_model(csv_path):

    df = pd.read_csv(csv_path)

    X = df[FEATURES]
    y = df['purchased']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_r, y_r = smote.fit_resample(X_train_scaled, y_train)

    model = LogisticRegression(max_iter=2000)
    model.fit(X_r, y_r)

    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "roc_auc": roc_auc_score(y_test, y_prob),
        "report": report
    }
