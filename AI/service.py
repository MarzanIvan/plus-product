import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from typing import Optional
import traceback
import boto3
from botocore.client import Config

app = FastAPI(title="predict")

class S3Client:
    def __init__(
        self,
        bucket: str,
        endpoint: Optional[str] = "https://storage.yandexcloud.net",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket = bucket

        self.s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=aws_access_key_id or os.getenv("YANDEX_ACCESS_KEY_ID"),
            aws_secret_access_key=aws_secret_access_key or os.getenv("YANDEX_SECRET_ACCESS_KEY"),
            config=Config(signature_version="s3v4"),
            region_name="ru-central1"
        )
    def read_json(self, key: str) -> pd.DataFrame:
        """Read JSON file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_json(io.BytesIO(obj["Body"].read()))

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read PARQUET file """
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return pd.read_parquet(io.BytesIO(obj["Body"].read()))

    def download_model(self, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)
        return local_path

    def upload_model(self, local_path: str, key: str):
        self.s3.upload_file(local_path, self.bucket, key)

    def upload_file(self, local_path: str, key: str):
        """Upload any file to S3."""
        self.s3.upload_file(local_path, self.bucket, key)

    def download_file(self, key: str, local_path: str):
        """Download any file """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.s3.download_file(self.bucket, key, local_path)

    def list_files(self, prefix: str = ""):
        """List objects in bucket."""
        response = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [item["Key"] for item in response["Contents"]]

OUTPUT_MODEL_PRODUCT = os.getenv("MODEL_CREDIT_PATH", "models/product/product.joblib")
S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

LOCAL_MODEL_DIR = "models"

s3 = S3Client(
    bucket=S3_BUCKET,
    aws_access_key_id=os.getenv("YANDEX_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("YANDEX_SECRET_ACCESS_KEY"),
)


def get_or_load_model(model_ref, local_path: str, model_type: str):
    if model_ref["model"] is not None:
        return model_ref["model"]

    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.exists(local_path):
        s3.download_file(local_path, local_path)

    if model_type == "keras":
        model_ref["model"] = load_model(local_path, compile=False)
    else:
        model_ref["model"] = joblib.load(local_path)

    return model_ref["model"]

credit_model_ref = {"model": None}
insurance_model_ref = {"model": None}
investment_model_ref = {"model": None}

def load_model_from_s3(s3_key: str, loader: str):
    local_path = os.path.join(LOCAL_MODEL_DIR, s3_key)

    s3.download_model(
        key=s3_key,
        local_path=local_path
    )

    if loader == "joblib":
        return joblib.load(local_path)

    if loader == "keras":
        return load_model(local_path)

    raise ValueError(f"Unsupported loader type: {loader}")


class ProductPredictRequest(BaseModel):
    products: str

class ProductPredictResponse(BaseModel):
    risk_level: str

@app.post("/predict/product", response_model=ProductPredictResponse)
def predict_credit(req: ProductPredictRequest):
    try:
        model = get_or_load_model(
            credit_model_ref,
            OUTPUT_MODEL_PRODUCT,
            "joblib"
        )
        pd_default = 0.2
        
        risk_level = (
            "LOW" if pd_default < 0.2
            else "MEDIUM" if pd_default < 0.5
            else "HIGH"
        )
        return ProductPredictResponse(
            risk_level=risk_level
        )
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Credit prediction failed")