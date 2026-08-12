from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
import mlflow.xgboost
import os
import logging

# Configure Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("churn-api")

class MockChurnModel:
    """Mock model implementing the XGBoost/Sklearn prediction API interface for dry-runs."""
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Return 1 if recency > 60 else 0
        return np.where(df['Recency'] > 60, 1, 0)
        
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        # Simulate probability of churn
        probs = np.where(df['Recency'] > 60, 0.85, 0.15)
        # Stack to replicate classifier return format [[1-p, p]]
        return np.stack([1 - probs, probs], axis=1)

# Global model variable
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Initializing application lifespan: Loading ML model...")
    try:
        # Searching for the latest run in mlruns/ (checking both root and src/analysis/)
        mlruns_dir_root = os.path.join(os.path.dirname(__file__), '../../mlruns')
        mlruns_dir_analysis = os.path.join(os.path.dirname(__file__), '../analysis/mlruns')
        
        mlruns_dir = mlruns_dir_root if os.path.exists(mlruns_dir_root) else mlruns_dir_analysis
        
        if os.path.exists(mlruns_dir):
            import glob
            # Try to find the xgboost_churn_model folder or models registry artifacts folder
            model_paths = glob.glob(os.path.join(mlruns_dir, "*/*/artifacts/xgboost_churn_model"))
            if not model_paths:
                model_paths = glob.glob(os.path.join(mlruns_dir, "*/models/*/artifacts"))
                
            if model_paths:
                latest_model_path = model_paths[-1] # Grabbing the latest logged artifact
                logger.info(f"Found model artifact path: {latest_model_path}. Loading...")
                model = mlflow.xgboost.load_model(latest_model_path)
                logger.info("Production MLflow model loaded successfully.")
            else:
                logger.warning("Warning: MLflow model artifact not found locally. Booting in Mock/Dry-Run mode.")
                model = MockChurnModel()
        else:
            logger.warning("Warning: MLflow runs directory not found. Booting in Mock/Dry-Run mode.")
            model = MockChurnModel()
    except Exception as e:
        logger.error(f"Error loading production model: {e}. Falling back to Mock/Dry-Run mode.", exc_info=True)
        model = MockChurnModel()
    
    yield
    logger.info("Shutting down application lifespan: Releasing resources...")

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Enterprise REST API for predicting Customer Churn using XGBoost",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic schema for input validation
class CustomerData(BaseModel):
    customer_id: str
    recency: float
    frequency: float
    monetary_value: float

class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: int
    risk_level: str

@app.get("/health")
def health_check():
    is_mock = isinstance(model, MockChurnModel)
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "mode": "mock" if is_mock else "production"
    }

@app.post("/predict", response_model=ChurnResponse)
def predict_churn(data: CustomerData):
    logger.info(f"Received prediction request for customer_id: {data.customer_id}")
    
    if model is None:
        logger.error("Inference failed: Model variable is uninitialized.")
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
        
    try:
        # Prepare input for classifier
        input_df = pd.DataFrame([{
            "Recency": data.recency,
            "Frequency": data.frequency,
            "MonetaryValue": data.monetary_value
        }])
        
        # Consistent API calls regardless of mock vs production model
        prob = float(model.predict_proba(input_df)[0][1])
        pred = int(model.predict(input_df)[0])
        
        risk_level = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        
        logger.info(f"Prediction processed for customer_id {data.customer_id}. Probability={prob:.4f}, Prediction={pred}, Risk={risk_level}")
        
        return ChurnResponse(
            customer_id=data.customer_id,
            churn_probability=prob,
            churn_prediction=pred,
            risk_level=risk_level
        )
    except Exception as e:
        logger.error(f"Inference error for customer_id {data.customer_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference process failure: {str(e)}")

