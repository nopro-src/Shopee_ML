# app/routers/predict_router.py
from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
from app.services.predict_service import predict_product

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")


@router.get("/predict-product-page")
def page_predict(request: Request):
    return templates.TemplateResponse("predict_product.html", {"request": request})


@router.post("/predict-product")
async def api_predict(payload: dict):
    return predict_product(payload)
