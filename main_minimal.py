import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Create app
app = FastAPI(title="Basketball Referee API - Railway Test")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "message": "🏀 Basketball Referee API on Railway",
        "status": "running",
        "python_version": sys.version,
        "port": os.getenv("PORT", "not set"),
        "railway_env": os.getenv("RAILWAY_ENVIRONMENT", "not set"),
        "endpoints": {
            "root": "/",
            "health": "/health",
            "env": "/env"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/env")
async def env():
    """Show environment info for debugging"""
    return {
        "port": os.getenv("PORT"),
        "railway_environment": os.getenv("RAILWAY_ENVIRONMENT"),
        "railway_static_url": os.getenv("RAILWAY_STATIC_URL"),
        "model_url_set": bool(os.getenv("MODEL_URL")),
        "working_dir": os.getcwd(),
        "python_version": sys.version
    }


if __name__ == "__main__":
    import uvicorn

    # MUST use Railway's PORT
    port = int(os.getenv("PORT", 8000))

    print(f"Starting on port {port}")
    print(f"Railway URL")