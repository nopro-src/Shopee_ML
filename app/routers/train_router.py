# app/routers/train_router.py
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from app.services.train_service import train_product_model
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
    # Save uploaded file
    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    # Train and get metrics
    metrics = train_product_model(path)

    # Render same template but with metrics
    return templates.TemplateResponse("upload_train.html", {
        "request": request,
        "metrics": metrics,
        "filename": file.filename
    })
