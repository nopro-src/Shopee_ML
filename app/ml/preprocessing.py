import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_product(df: pd.DataFrame):
    encoder = LabelEncoder()
    df["category_encoded"] = encoder.fit_transform(df["category"])
    X = df[["price", "quantity", "category_encoded"]]
    y = df["ordered"]
    return X, y, encoder

def preprocess_customer(df: pd.DataFrame):
    encoder = LabelEncoder()
    df["segment_encoded"] = encoder.fit_transform(df["segment"])
    X = df[["age", "income", "segment_encoded"]]
    y = df["label"]
    return X, y, encoder
