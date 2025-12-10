from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from app.services.train_service import train_product_model, train_customer_model
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "app/static/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.get("/upload-train")
def page_upload(request: Request):
    return templates.TemplateResponse("upload_train.html", {"request": request})

@router.post("/train-product")
async def train_product(request: Request, file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    metrics = train_product_model(path)
    return templates.TemplateResponse("upload_train.html", {
        "request": request,
        "metrics": metrics,
        "filename": file.filename,
        "trained_type": "product"
    })

@router.post("/train-customer")
async def train_customer(request: Request, file: UploadFile = File(...)):
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    metrics = train_customer_model(path, n_clusters=5)
    return templates.TemplateResponse("upload_train.html", {
        "request": request,
        "metrics": metrics,
        "filename": file.filename,
        "trained_type": "customer"
    })
