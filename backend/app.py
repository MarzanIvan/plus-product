from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import date
import httpx
import os


ML_URL = os.getenv("ML_URL", "http://ml_service:9000") # access to datasets and models
AI_URL = os.getenv("AI_URL", "http://predict_service:7000") # access to models

app = FastAPI(title="api")

"""
    AI MODULE
    communication to http://predict_service:7000
"""

class ProductInput(BaseModel):
    products: str
    

class ProductPredictResponse(BaseModel):
    risk_level: str              # LOW / MEDIUM / HIGH

@app.post("/analysis_credit/",response_model=ProductPredictResponse)
async def credit_risk(data: ProductInput):
    return await call_predict(
        "predict/credit",
        data.dict()
    )

async def call_predict(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            response = await client.post(f"{AI_URL}/{route}", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

"""
    ML MODULE
    communication to http://ml_service:9000
"""
class TrainRequest(BaseModel):
    dataset_name: str
    dataset_path: str


async def call_training(route: str, payload: dict):
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            response = await client.post(
                f"{ML_URL}/{route}",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/ml_product")
async def train_credit(req: TrainRequest):
    return await call_training("train/product", req.dict())

@app.get("/health")
def health():
    return {"status": "ok", "service": "backend"}
