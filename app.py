import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import joblib

app = FastAPI()

model = joblib.load("titanic.pkl")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_post(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})


@app.post("/predict_csv")
async def predict_csv(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    preds = model.predict(df)

    safe_preds = [None if (isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)))
                  else float(x)
                  for x in preds]

    results = [{"index": i, "predicted": p} for i, p in enumerate(safe_preds)]

    return {"predictions": results}
