from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.routers import train_router, predict_router, csv_router

app = FastAPI()

# Static + Template
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Routers
app.include_router(csv_router.router)
app.include_router(train_router.router)
app.include_router(predict_router.router)
