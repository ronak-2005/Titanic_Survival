from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Titanic Survival Prediction")
MAX_ROWS = 10000  

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model paths
MODEL_PATH_RENDER = "/mnt/data/titanic.pkl"
MODEL_PATH_LOCAL = os.path.join(BASE_DIR, "titanic.pkl")

# Load model
try:
    if os.path.exists(MODEL_PATH_RENDER):
        model_path = MODEL_PATH_RENDER
        logger.info("Using Render path for model file")
    elif os.path.exists(MODEL_PATH_LOCAL):
        model_path = MODEL_PATH_LOCAL
        logger.info("Using local path for model file")
    else:
        raise FileNotFoundError("Model file not found.")

    model = joblib.load(model_path)
    logger.info("Model loaded successfully")
    
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise e

# Setup templates
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main HTML page"""
    try:
        return templates.TemplateResponse("form_titanic.html", {"request": request})
    except Exception as e:
        logger.error(f"Error serving home page: {str(e)}")
        return HTMLResponse("""
        <!DOCTYPE html>
        <html><head><title>Titanic Survival Predictor</title></head>
        <body>
        <h1>Titanic Survival Predictor</h1>
        <form action="/predict_csv" method="post" enctype="multipart/form-data">
            <input type="file" name="file" accept=".csv" required>
            <button type="submit">Upload & Predict</button>
        </form>
        </body></html>
        """)

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    """Handle CSV upload and return survival predictions as JSON"""
    
    logger.info(f"Received file: {file.filename}")
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Please upload a CSV file")
        
        # Read CSV
        try:
            df = pd.read_csv(file.file, nrows=MAX_ROWS)
            logger.info(f"CSV loaded. Shape: {df.shape}")
            logger.info(f"CSV columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Make predictions
        try:
            preds = model.predict(df)
            logger.info(f"Predictions completed. Count: {len(preds)}")
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error making predictions: {str(e)}")
        
        # Clean predictions (handle NaN/inf values)
        try:
            safe_preds = []
            for x in preds:
                if isinstance(x, (float, np.floating)) and (np.isnan(x) or np.isinf(x)):
                    safe_preds.append(None)
                elif isinstance(x, str):
                    safe_preds.append(x)
                else:
                    safe_preds.append(int(x))
            
            logger.info(f"Predictions cleaned. Valid predictions: {sum(1 for p in safe_preds if p is not None)}")
        except Exception as e:
            logger.error(f"Error cleaning predictions: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing predictions: {str(e)}")
        
        # Prepare all results
        all_results = []
        for i, p in enumerate(safe_preds):
            if p is None:
                prediction_label = "Unknown"
            else:
                prediction_label = "Survived" if p == 1 else "Not Survived"
            
            all_results.append({
                "index": i,
                "prediction_label": prediction_label
            })
        
        logger.info(f"Returning all {len(all_results)} predictions")
        
        return {
            "status": "success",
            "predictions": all_results,
            "total_rows_processed": len(all_results)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)