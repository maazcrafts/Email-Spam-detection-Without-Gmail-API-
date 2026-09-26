"""Compatibility entry point for the production web application.

The application is now served by FastAPI instead of Streamlit so users only
need a web browser. The actual API and frontend live in app.py and web/.
"""

import os

import uvicorn

from app import app


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )
