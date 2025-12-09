# app/routers/csv_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import os

router = APIRouter()
UPLOAD_DIR = "app/static/uploads"
_cached = {}  # { filename: df }  -- đơn giản, lưu tạm trong memory

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

@router.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    path = os.path.join(UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    # load header + small preview to verify (no full rendering)
    try:
        # load but not necessarily whole file; we'll cache full df for simplicity
        df = pd.read_csv(path, dtype=str)  # read as strings to avoid type issues
        _cached[file.filename] = df
        return JSONResponse({"filename": file.filename, "rows": len(df), "cols": list(df.columns)})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot read CSV: {e}")

@router.get("/load-csv-chunk")
def load_csv_chunk(filename: str, start: int = 0, size: int = 100):
    """
    Return a chunk of rows as JSON records.
    start: zero-based start index
    size: number of rows
    """
    if filename not in _cached:
        raise HTTPException(status_code=404, detail="File not cached. Upload first.")
    df = _cached[filename]
    end = min(start + size, len(df))
    records = df.iloc[start:end].fillna("").to_dict(orient="records")
    return {"data": records, "start": start, "end": end, "total": len(df)}

@router.get("/get-row")
def get_row(filename: str, index: int):
    """Return single row (dict) for given index."""
    if filename not in _cached:
        raise HTTPException(status_code=404, detail="File not cached.")
    df = _cached[filename]
    if index < 0 or index >= len(df):
        raise HTTPException(status_code=400, detail="Index out of range.")
    row = df.iloc[index].fillna("").to_dict()
    return row
