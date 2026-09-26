from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
from contextlib import asynccontextmanager
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path
from typing import Any

import pandas as pd
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "completeSpamAssassin.csv"
WEB_DIR = BASE_DIR / "web"

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_TOKEN_COOKIE = "gmail_session"
GMAIL_STATE_COOKIE = "gmail_oauth_state"

ml: dict[str, Any] = {}


def normalize_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = pd.to_numeric(series, errors="coerce")
        unique = set(values.dropna().astype(int).unique().tolist())
        if unique.issubset({0, 1}):
            return values.fillna(0).astype(int)
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

    vectorizer = CountVectorizer(stop_words="english", max_features=50000, min_df=1)
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
    ml.update(train_model())
    yield
    ml.clear()


app = FastAPI(
    title="Gmail Spam Detection API",
    description="NLP spam classification with optional Gmail OAuth integration.",
    version="3.0.0",
    lifespan=lifespan,
)

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


def gmail_configured() -> bool:
    return bool(os.getenv("GMAIL_CLIENT_ID") and os.getenv("GMAIL_CLIENT_SECRET"))


def session_fernet() -> Fernet:
    secret = os.getenv("GMAIL_SESSION_SECRET")
    if not secret:
        raise HTTPException(
            status_code=500,
            detail="GMAIL_SESSION_SECRET is not configured in Vercel.",
        )
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


def sign_state(state: str) -> str:
    secret = os.getenv("GMAIL_SESSION_SECRET", "")
    signature = hmac.new(secret.encode(), state.encode(), hashlib.sha256).hexdigest()
    return f"{state}.{signature}"


def verify_state(value: str | None) -> str | None:
    if not value or "." not in value:
        return None
    state, signature = value.rsplit(".", 1)
    expected = hmac.new(
        os.getenv("GMAIL_SESSION_SECRET", "").encode(),
        state.encode(),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return None
    return state


def client_config() -> dict[str, Any]:
    if not gmail_configured():
        raise HTTPException(
            status_code=503,
            detail="Gmail OAuth is not configured. Add GMAIL_CLIENT_ID and GMAIL_CLIENT_SECRET.",
        )

    return {
        "web": {
            "client_id": os.environ["GMAIL_CLIENT_ID"],
            "client_secret": os.environ["GMAIL_CLIENT_SECRET"],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }


def redirect_uri(request: Request) -> str:
    configured = os.getenv("GMAIL_REDIRECT_URI")
    if configured:
        return configured.rstrip("/")
    return str(request.base_url).rstrip("/") + "/api/gmail/callback"


def create_flow(request: Request, state: str | None = None) -> Flow:
    flow = Flow.from_client_config(client_config(), scopes=GMAIL_SCOPES, state=state)
    flow.redirect_uri = redirect_uri(request)
    return flow


def store_refresh_token(response: RedirectResponse, refresh_token: str) -> None:
    encrypted = session_fernet().encrypt(refresh_token.encode()).decode()
    response.set_cookie(
        GMAIL_TOKEN_COOKIE,
        encrypted,
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )


def get_refresh_token(request: Request) -> str | None:
    encrypted = request.cookies.get(GMAIL_TOKEN_COOKIE)
    if not encrypted:
        return None
    try:
        return session_fernet().decrypt(encrypted.encode()).decode()
    except InvalidToken:
        return None


def gmail_credentials(request: Request) -> Credentials:
    refresh_token = get_refresh_token(request)
    if not refresh_token:
        raise HTTPException(status_code=401, detail="Gmail is not connected.")

    credentials = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GMAIL_CLIENT_ID"],
        client_secret=os.environ["GMAIL_CLIENT_SECRET"],
        scopes=GMAIL_SCOPES,
    )
    try:
        credentials.refresh(GoogleRequest())
    except Exception as exc:
        raise HTTPException(
            status_code=401,
            detail="Gmail authorization expired or was revoked. Connect Gmail again.",
        ) from exc
    return credentials


def header_value(headers: list[dict[str, str]], name: str) -> str:
    wanted = name.lower()
    for header in headers:
        if header.get("name", "").lower() == wanted:
            return header.get("value", "")
    return ""


def decode_part(data: str) -> str:
    raw = base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))
    return raw.decode("utf-8", errors="replace")


def extract_body(payload: dict[str, Any]) -> str:
    mime = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")
    if body_data and (mime.startswith("text/plain") or not payload.get("parts")):
        return decode_part(body_data)

    for part in payload.get("parts", []):
        text = extract_body(part)
        if text.strip():
            return text

    return ""


def clean_html(text: str) -> str:
    import re
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return unescape(re.sub(r"\s+", " ", text)).strip()


def classify_text(text: str) -> dict[str, Any]:
    vector = ml["vectorizer"].transform([text])
    model = ml["model"]
    prediction = int(model.predict(vector)[0])
    probabilities = model.predict_proba(vector)[0]
    return {
        "is_spam": prediction == 1,
        "prediction": "Spam" if prediction == 1 else "Not Spam",
        "confidence": round(float(max(probabilities)) * 100, 2),
    }


def message_to_result(message: dict[str, Any]) -> dict[str, Any]:
    payload = message.get("payload", {})
    headers = payload.get("headers", [])
    subject = header_value(headers, "Subject") or "(No subject)"
    sender = header_value(headers, "From") or "Unknown sender"
    date_value = header_value(headers, "Date")
    body = extract_body(payload)
    text = f"Subject: {subject}\nFrom: {sender}\n{clean_html(body)}"
    result = classify_text(text)

    try:
        timestamp = parsedate_to_datetime(date_value).isoformat() if date_value else None
    except (TypeError, ValueError):
        timestamp = date_value or None

    return {
        "id": message.get("id"),
        "thread_id": message.get("threadId"),
        "subject": subject,
        "from": sender,
        "date": timestamp,
        "snippet": message.get("snippet", ""),
        **result,
    }


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": "model" in ml,
        "gmail_configured": gmail_configured(),
        "records": ml.get("records", 0),
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
    return classify_text(payload.email.strip())


@app.get("/api/gmail/status")
async def gmail_status(request: Request) -> dict[str, Any]:
    connected = bool(get_refresh_token(request))
    return {"configured": gmail_configured(), "connected": connected}


@app.get("/api/gmail/login")
async def gmail_login(request: Request):
    if not gmail_configured() or not os.getenv("GMAIL_SESSION_SECRET"):
        raise HTTPException(
            status_code=503,
            detail="Configure GMAIL_CLIENT_ID, GMAIL_CLIENT_SECRET and GMAIL_SESSION_SECRET in Vercel.",
        )

    state = secrets.token_urlsafe(32)
    flow = create_flow(request, state=state)
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    response = RedirectResponse(authorization_url, status_code=302)
    response.set_cookie(
        GMAIL_STATE_COOKIE,
        sign_state(state),
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=600,
        path="/",
    )
    return response


@app.get("/api/gmail/callback")
async def gmail_callback(request: Request, code: str | None = None, state: str | None = None, error: str | None = None):
    if error:
        return RedirectResponse("/?gmail=denied", status_code=302)

    expected_state = verify_state(request.cookies.get(GMAIL_STATE_COOKIE))
    if not code or not state or not expected_state or not hmac.compare_digest(state, expected_state):
        raise HTTPException(status_code=400, detail="Invalid Gmail OAuth state.")

    flow = create_flow(request, state=state)
    try:
        flow.fetch_token(code=code)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Google OAuth authorization failed.") from exc

    refresh_token = flow.credentials.refresh_token
    if not refresh_token:
        raise HTTPException(
            status_code=400,
            detail="Google did not return a refresh token. Reconnect Gmail and grant access.",
        )

    response = RedirectResponse("/?gmail=connected", status_code=302)
    store_refresh_token(response, refresh_token)
    response.delete_cookie(GMAIL_STATE_COOKIE, path="/")
    return response


@app.get("/api/gmail/messages")
async def gmail_messages(request: Request, limit: int = 20):
    limit = max(1, min(limit, 50))
    credentials = gmail_credentials(request)

    try:
        service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
        listing = (
            service.users()
            .messages()
            .list(userId="me", maxResults=limit, q="in:inbox")
            .execute()
        )
        messages = []
        for item in listing.get("messages", []):
            message = (
                service.users()
                .messages()
                .get(userId="me", id=item["id"], format="full")
                .execute()
            )
            messages.append(message_to_result(message))

        spam_count = sum(item["is_spam"] for item in messages)
        return {
            "messages": messages,
            "total": len(messages),
            "spam_count": spam_count,
            "safe_count": len(messages) - spam_count,
        }
    except HttpError as exc:
        if getattr(exc.resp, "status", None) == 401:
            raise HTTPException(status_code=401, detail="Gmail authorization expired. Reconnect Gmail.") from exc
        raise HTTPException(status_code=502, detail="Gmail API request failed.") from exc


@app.post("/api/gmail/disconnect")
async def gmail_disconnect():
    response = JSONResponse({"connected": False})
    response.delete_cookie(GMAIL_TOKEN_COOKIE, path="/")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
