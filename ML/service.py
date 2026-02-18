import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


from s3_client import S3Client

OUTPUT_MODEL_PRODUCT = os.getenv("MODEL_CREDIT_PATH", "models/product/product.joblib")
S3_BUCKET = os.getenv("S3_BUCKET", "risk-model-storage")

app = FastAPI(title="ML Service")

s3 = S3Client(
    bucket=S3_BUCKET,
    aws_access_key_id=os.getenv("YANDEX_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("YANDEX_SECRET_ACCESS_KEY"),
)

def load_dataset(s3_path: str, *, csv_sep: str = ";"):
    if s3_path.endswith(".csv"):
        return s3.read_csv(
            s3_path,
            sep=csv_sep
        )
    elif s3_path.endswith(".json"):
        return s3.read_json(s3_path)
    elif s3_path.endswith(".parquet"):
        return s3.read_parquet(s3_path)
    else:
        raise ValueError(f"Unsupported dataset format: {s3_path}")


class TrainRequest(BaseModel):
    dataset_name: str
    dataset_path: str

@app.post("/train/product")
def train_credit(req: TrainRequest):
    try:
        return train_product_model(req.dataset_name, req.dataset_path)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(tb)
        raise HTTPException(500, tb)


@app.get("/health")
def health():
    return {"status": "ok", "service": "training-service"}

def train_product_model(dataset_name: str, dataset_path: str):
    df = load_dataset(
        dataset_path,
        csv_sep=","
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])

    local_dir = "models/product"
    os.makedirs(local_dir, exist_ok=True)

    model_filename = "product.joblib"
    local_path = os.path.join(local_dir, model_filename)
    s3_key = f"models/product/{model_filename}"

    joblib.dump(pipeline, local_path)

    s3.upload_file(
        local_path=local_path,
        key=s3_key
    )

    return {
        "model_path": local_path,
        "s3_key": s3_key
    }
