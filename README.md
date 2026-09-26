# 📧 Gmail Spam Detection

A browser-based machine-learning application that can classify pasted email text **and scan recent Gmail inbox messages** using the Gmail API.

## What it does

- 🔐 Google OAuth 2.0 login
- 📬 Reads recent **Inbox** messages through Gmail API
- 🤖 Classifies each message as **Spam** or **Not Spam**
- 📊 Shows model confidence for every message
- ✍️ Still supports manual email classification
- 🔒 Requests the Gmail **read-only** scope only
- 🌐 Designed for Vercel deployment
- 🧩 FastAPI backend + responsive HTML/CSS/JavaScript frontend
- 🐳 Docker is not required

The Gmail integration does **not** send, delete, modify, or label messages.

## Architecture

```text
Browser
   │
   ├── Connect Gmail
   │       │
   │       ▼
   │   Google OAuth 2.0
   │       │
   │       ▼
   │   Encrypted refresh-token cookie
   │
   └── Scan Inbox
           │
           ▼
      FastAPI on Vercel
           │
           ├── Gmail API (read-only)
           │
           └── NLP classifier
                 │
                 ├── CountVectorizer
                 └── Logistic Regression
```

Vercel currently supports FastAPI with zero configuration, and FastAPI static files can also be served through Vercel's CDN. urlVercel FastAPI documentationhttps://vercel.com/kb/fastapi

## 🧠 Machine Learning

The classifier uses:

- **CountVectorizer**
- **Logistic Regression**
- **completeSpamAssassin.csv**
- 80/20 stratified train/test split

The model is trained when the application starts and reused for predictions.

## 🔐 Gmail OAuth setup

Google requires OAuth 2.0 authorization for an application accessing a user's Gmail data. This project requests:

```text
https://www.googleapis.com/auth/gmail.readonly
```

Google documents this as the read-only Gmail scope and recommends a server-side OAuth flow for web applications. citeturn0search0turn0search4

### 1. Create a Google Cloud project

Open urlGoogle Cloud Consolehttps://console.cloud.google.com/.

Create/select a project and enable:

**Gmail API**

Then configure the Google OAuth consent screen.

### 2. Create OAuth credentials

In Google Cloud:

**Google Auth Platform → Clients → Create Client**

Choose:

**Web application**

Add this Authorized redirect URI:

```text
https://YOUR-VERCEL-DOMAIN.vercel.app/api/gmail/callback
```

For local testing you can also register:

```text
http://localhost:8000/api/gmail/callback
```

Google requires the redirect URI used by the application to be registered with the OAuth client. citeturn0search2

### 3. Add Vercel environment variables

In your Vercel project:

**Settings → Environment Variables**

Add:

```text
GMAIL_CLIENT_ID=your-google-client-id
GMAIL_CLIENT_SECRET=your-google-client-secret
GMAIL_SESSION_SECRET=a-long-random-secret
GMAIL_REDIRECT_URI=https://YOUR-VERCEL-DOMAIN.vercel.app/api/gmail/callback
```

Do **not** commit the client secret or refresh tokens to GitHub.

### 4. Deploy

Import this GitHub repository into Vercel and deploy it.

No Docker configuration is required. Vercel recognizes the root FastAPI application automatically. citeturn1search0turn1search1

After deployment:

1. Open your Vercel URL.
2. Click **Connect Gmail**.
3. Sign in with Google.
4. Grant read-only Gmail access.
5. Click **Scan Inbox**.
6. The application fetches recent inbox messages and classifies them.

## Local development

Install:

```bash
pip install -r requirements.txt
```

Create a local `.env` using `.env.example` and set:

```text
GMAIL_REDIRECT_URI=http://localhost:8000/api/gmail/callback
```

Then run:

```bash
uvicorn app:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

## API

```text
GET  /api/health
GET  /api/stats
POST /api/predict

GET  /api/gmail/status
GET  /api/gmail/login
GET  /api/gmail/callback
GET  /api/gmail/messages
POST /api/gmail/disconnect
```

FastAPI's interactive documentation is available at:

```text
/docs
```

## 📁 Project structure

```text
.
├── app.py
├── completeSpamAssassin.csv
├── email-spam-detection.py
├── requirements.txt
├── .env.example
├── .python-version
└── web/
    ├── index.html
    ├── styles.css
    └── app.js
```

## 🔒 Security and privacy

- Gmail access uses Google's OAuth 2.0 flow.
- The app requests `gmail.readonly`, not send/delete/modify permissions.
- The refresh token is encrypted before being placed in an HttpOnly, Secure cookie.
- OAuth state is signed to reduce CSRF risk.
- Client secrets belong in Vercel environment variables, never GitHub.
- The classifier only processes messages returned from the user's inbox.
- The app does not automatically alter Gmail labels or message state.

For Google's OAuth implementation guidance, see urlGmail API OAuth documentationhttps://developers.google.com/workspace/gmail/api/auth/web-server.

## ⚠️ Important limitations

This is an ML classifier, not Gmail's native spam engine. A **Spam** result means this project's model predicts the message is spam; it does not mean Google has marked the message as spam.

The first model initialization can take time because the training dataset is loaded and the model is trained on the server.

Vercel supports large Python Functions up to 5 GB through its current Large Functions support, which is relevant because scientific Python dependencies can be large. citeturn1search10

---

Built with Python, FastAPI, Gmail API, Pandas, NumPy and Scikit-learn.
