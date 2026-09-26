from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "completeSpamAssassin.csv"
WEB_DIR = BASE_DIR / "web"

ml: dict[str, Any] = {}


def normalize_labels(series: pd.Series) -> pd.Series:
    """Convert common spam/ham label formats into binary 0/1 labels."""
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        unique = set(values.dropna().astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return values.fillna(0).astype(int)
        # Fall back to a deterministic binary mapping for other numeric labels.
        return (values == values.max()).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    spam_values = {"1", "spam", "junk", "yes", "true", "malicious"}
    return normalized.isin(spam_values).astype(int)


def train_model() -> dict[str, Any]:
    if not DATASET_PATH.exists():
        raise RuntimeError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required = {"Body", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["Body", "Label"]).copy()
    df["Body"] = df["Body"].astype(str)
    df["Target"] = normalize_labels(df["Label"])

    if df["Target"].nunique() < 2:
        raise RuntimeError("Dataset must contain both spam and non-spam examples.")

    vectorizer = CountVectorizer(
        stop_words="english",
        max_features=50000,
        min_df=1,
    )
    features = vectorizer.fit_transform(df["Body"])

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        df["Target"],
        test_size=0.20,
        random_state=42,
        stratify=df["Target"],
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    return {
        "model": model,
        "vectorizer": vectorizer,
        "records": len(df),
        "features": features.shape[1],
        "training_emails": len(y_train),
        "testing_emails": len(y_test),
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
    }


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Train once when the server starts, then reuse the model for every request.
    ml.update(train_model())
    yield
    ml.clear()


app = FastAPI(
    title="Email Spam Detection API",
    description="Web API for spam classification using CountVectorizer and Logistic Regression.",
    version="2.0.0",
    lifespan=lifespan,
)

# The frontend is served by the same FastAPI process in production.
# CORS is also enabled so the API can be called from GitHub Pages or another
# static frontend later without changing the backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


class PredictionRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=50000)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": "model" in ml,
        "records": ml.get("records", 0),
        "features": ml.get("features", 0),
    }


@app.get("/api/stats")
async def stats() -> dict[str, Any]:
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    return {
        "records": ml["records"],
        "features": ml["features"],
        "training_emails": ml["training_emails"],
        "testing_emails": ml["testing_emails"],
        "accuracy": round(ml["accuracy"] * 100, 2),
    }


@app.post("/api/predict")
async def predict(payload: PredictionRequest) -> dict[str, Any]:
    if "model" not in ml:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    text = payload.email.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Email message cannot be empty.")

    vector = ml["vectorizer"].transform([text])
    model = ml["model"]

    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    confidence = float(max(probabilities) * 100)

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "is_spam": prediction == 1,
        "confidence": round(confidence, 2),
    }


if __name__ == "__main__":
    import os
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
