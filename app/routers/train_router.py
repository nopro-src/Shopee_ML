from fastapi import APIRouter, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from app.services.train_service import train_product_model
import os

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "app/static/uploads"


@router.get("/upload-train")
def page_upload(request: Request):
    return templates.TemplateResponse("upload_train.html", {"request": request})


@router.post("/train-product")
async def train_product(file: UploadFile = File(...)):

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = train_product_model(file_path)

    return {
        "message": "Train thành công!",
        "metrics": result
    }
