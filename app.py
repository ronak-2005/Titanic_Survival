from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(title="Titanic Survival Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load(os.path.join(BASE_DIR, "titanic.pkl"))

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("form_titanic.html", {"request": request})

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    df = pd.read_csv(file.file, nrows=MAX_ROWS)
    preds = model.predict(df)

    safe_preds = [
        None if (isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)))
        else int(x) if not isinstance(x, str) else x
        for x in preds
    ]

    results = [
        {"index": i, "prediction_label": "Survived" if p == 1 else "Not Survived"}
        for i, p in enumerate(safe_preds)
    ]

    return {
        "predictions": results,
        "rows_processed": len(df)
    }
