from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "completeSpamAssassin.csv"
WEB = ROOT / "web"

app = FastAPI(
    title="Email Spam Detection API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_ml: dict[str, Any] | None = None


class PredictionRequest(BaseModel):
    email: str = Field(..., min_length=1, max_length=50000)


def load_model() -> dict[str, Any]:
    global _ml

    if _ml is not None:
        return _ml

    if not DATASET.exists():
        raise RuntimeError("completeSpamAssassin.csv was not found.")

    df = pd.read_csv(DATASET)

    required = {"Body", "Label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing: {sorted(missing)}")

    df = df.dropna(subset=["Body", "Label"]).copy()
    df["Body"] = df["Body"].astype(str)

    labels = df["Label"]
    if pd.api.types.is_numeric_dtype(labels):
        target = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
        if not set(target.unique()).issubset({0, 1}):
            target = (target == target.max()).astype(int)
    else:
        target = labels.astype(str).str.strip().str.lower().isin(
            {"1", "spam", "junk", "yes", "true", "malicious"}
        ).astype(int)

    vectorizer = CountVectorizer(stop_words="english", max_features=50000)
    x = vectorizer.fit_transform(df["Body"])

    x_train, x_test, y_train, y_test = train_test_split(
        x, target, test_size=0.20, random_state=42, stratify=target
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(x_test))

    _ml = {
        "model": model,
        "vectorizer": vectorizer,
        "records": len(df),
        "features": x.shape[1],
        "training": len(y_train),
        "testing": len(y_test),
        "accuracy": round(float(accuracy) * 100, 2),
    }
    return _ml


def get_model() -> dict[str, Any]:
    try:
        return load_model()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/", response_class=HTMLResponse)
async def home():
    return (WEB / "index.html").read_text(encoding="utf-8")


@app.get("/assets/{filename}")
async def assets(filename: str):
    allowed = {"styles.css": "text/css", "app.js": "application/javascript"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Asset not found")
    return Response(
        (WEB / filename).read_text(encoding="utf-8"),
        media_type=allowed[filename],
    )


@app.get("/api/health")
async def health():
    try:
        model = get_model()
        return {"status": "ok", "model_ready": True, "records": model["records"]}
    except HTTPException:
        raise


@app.get("/api/stats")
async def stats():
    model = get_model()
    return {
        "records": model["records"],
        "features": model["features"],
        "training_emails": model["training"],
        "testing_emails": model["testing"],
        "accuracy": model["accuracy"],
    }


@app.post("/api/predict")
async def predict(payload: PredictionRequest):
    text = payload.email.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Email message cannot be empty.")

    model = get_model()
    vector = model["vectorizer"].transform([text])
    classifier = model["model"]

    prediction = int(classifier.predict(vector)[0])
    probabilities = classifier.predict_proba(vector)[0]

    return {
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "is_spam": prediction == 1,
        "confidence": round(float(max(probabilities)) * 100, 2),
    }
