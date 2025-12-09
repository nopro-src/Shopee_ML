from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import JSONResponse
import pandas as pd
import os
from app.services.predict_service import predict_product

router = APIRouter()
templates = Jinja2Templates(directory="app/templates")

UPLOAD_DIR = "app/static/uploads"
CHUNK_SIZE = 200


@router.get("/predict-product-page")
def page_predict(request: Request):
    return templates.TemplateResponse("predict_product.html", {"request": request})


@router.post("/upload-csv-preview")
async def preview_csv(request: Request, file: UploadFile = File(...)):
    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR)

    path = f"{UPLOAD_DIR}/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())

    df = pd.read_csv(path, nrows=CHUNK_SIZE)
    df_html = df.to_html(classes="table table-striped", index=True)

    return templates.TemplateResponse("predict_product.html", {
        "request": request,
        "table": df_html,
        "filename": file.filename,
        "loaded_rows": CHUNK_SIZE
    })


@router.post("/load-more-rows")
async def load_more(payload: dict):
    filename = payload["filename"]
    offset = payload["offset"]

    path = f"{UPLOAD_DIR}/{filename}"
    df = pd.read_csv(path, skiprows=range(1, offset + 1), nrows=CHUNK_SIZE)

    if df.empty:
        return JSONResponse({"html": "", "done": True})

    html = df.to_html(classes="table table-striped", index=False)
    return JSONResponse({"html": html, "done": False})


@router.post("/select-row")
async def select_row(payload: dict):
    filename = payload["filename"]
    row_index = payload["row_index"]

    path = f"{UPLOAD_DIR}/{filename}"
    df = pd.read_csv(path)

    return df.iloc[int(row_index)].to_dict()


@router.post("/predict-product")
async def predict(data: dict):
    try:
        result = predict_product(data)
        return result
    except Exception as e:
        return {"error": str(e)}
