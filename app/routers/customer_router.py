# app/routers/customer_router.py
from fastapi import APIRouter, Request, HTTPException
from fastapi.templating import Jinja2Templates
import pandas as pd
import os
from app.services.cluster_labeling import label_cluster_from_summary

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

MODEL_DIR = "app/ml"
CUSTOMER_CLUSTER_DF_PATH = os.path.join(MODEL_DIR, "customer_clusters.parquet")


@router.get("/customer-page")
def page_customer(request: Request):
    return templates.TemplateResponse("customer_page.html", {"request": request})


@router.post("/get-user-cluster")
def get_user_cluster(payload: dict):
    """
    payload: {"user_id": "<id>"}
    Return: user_row, cluster_id, cluster_summary, cluster_name, cluster_description
    """
    if not os.path.exists(CUSTOMER_CLUSTER_DF_PATH):
        raise HTTPException(status_code=404, detail="Customer clustering model not found. Train first.")

    df = pd.read_parquet(CUSTOMER_CLUSTER_DF_PATH)
    user_id = payload.get("user_id")
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id required")

    # convert both sides to str for robust match
    df['user_id_str'] = df['user_id'].astype(str)
    found = df[df['user_id_str'] == str(user_id)]
    if found.empty:
        raise HTTPException(status_code=404, detail="User not found in cluster table")

    row = found.iloc[0].to_dict()
    cluster = int(row['cluster'])

    # compute cluster summary (mean) from the saved df
    cluster_df = df[df['cluster'] == cluster]
    # drop helper
    cluster_summary = cluster_df[['searched', 'in_cart', 'purchased', 'qty', 'total_price']].mean().to_dict()

    # get label + description
    cluster_name, cluster_desc = label_cluster_from_summary(cluster_summary)

    return {
        "user_row": row,
        "cluster": cluster,
        "cluster_name": cluster_name,
        "cluster_description": cluster_desc,
        "cluster_summary": cluster_summary
    }
