# 📧 Email Spam Detection

A browser-based machine-learning application that classifies an email as **Spam** or **Not Spam** using **NLP, CountVectorizer, and Logistic Regression**.

## Live architecture

The project no longer requires Python, Streamlit, Docker, or ML libraries on the user's device.

```text
Browser
   │
   ▼
FastAPI Web App
   │
   ├── HTML / CSS / JavaScript
   │
   └── /api/predict
          │
          ▼
   CountVectorizer
          │
          ▼
   Logistic Regression
          │
          ▼
   Spam / Not Spam + confidence
```

FastAPI serves the frontend and prediction API from one web service. The ML model is trained once when the server starts and reused for incoming requests.

## ✨ Features

- Paste an email directly into the browser
- Spam / Not Spam classification
- Prediction confidence
- Dataset and model statistics
- Confusion-matrix-backed test accuracy
- Responsive interface for desktop and mobile
- API health endpoint
- Interactive FastAPI API documentation at `/docs`
- No Gmail API, OAuth, inbox access, or user email storage
- No software installation required for end users

## 🧠 Machine Learning

The model uses:

- **CountVectorizer** for converting email text into numerical features
- **Logistic Regression** for binary classification
- **completeSpamAssassin.csv** as the training dataset
- An 80/20 stratified train/test split

The application normalizes common dataset label formats so the backend is not dependent on labels being stored as exactly `0/1`.

## 📁 Project Structure

```text
.
├── app.py
├── email-spam-detection.py
├── completeSpamAssassin.csv
├── requirements.txt
├── render.yaml
├── .python-version
└── web/
    ├── index.html
    ├── styles.css
    └── app.js
```

## 💻 Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start the application:

```bash
uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The API is available at:

```text
POST /api/predict
GET  /api/health
GET  /api/stats
```

## ☁️ Deploy to Render

The repository includes `render.yaml`, so it is prepared for a Render Web Service.

Use:

- **Runtime:** Python
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Health check:** `/api/health`

Render provides the public HTTPS URL for the application. The browser only communicates with the hosted web app; users do not need Python or Docker installed locally.

## 🐳 Docker

Docker is **not required** for this project. The deployment uses Render's native Python runtime.

A Docker deployment can be added later if the project needs a custom system environment.

## 🔐 Privacy

The current application only classifies text submitted to the prediction endpoint. It does not connect to Gmail and does not automatically read a user's inbox.

## ⚠️ Model limitation

Spam classification is a machine-learning prediction, not a guarantee. Confidence represents the model's predicted probability and should not be interpreted as certainty.

---

Built with Python, FastAPI, Pandas, NumPy and Scikit-learn.
