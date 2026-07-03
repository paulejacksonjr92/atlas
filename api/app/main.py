import os
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

APP_VERSION = "0.2.0"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://atlas-ollama:11434")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")

app = FastAPI(title="Atlas API", version=APP_VERSION)


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_id() -> str:
    return str(uuid4())


def with_metadata(payload: dict) -> dict:
    return {
        "request_id": request_id(),
        "created_at": utc_now(),
        **payload,
    }


def error_response(status_code: int, code: str, message: str, details=None) -> JSONResponse:
    payload = with_metadata(
        {
            "error": {
                "code": code,
                "message": message,
                "details": details,
            }
        }
    )
    return JSONResponse(status_code=status_code, content=payload)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return error_response(
        status_code=422,
        code="validation_error",
        message="Request validation failed.",
        details=exc.errors(),
    )


@app.exception_handler(httpx.HTTPStatusError)
async def upstream_status_exception_handler(request: Request, exc: httpx.HTTPStatusError):
    return error_response(
        status_code=502,
        code="upstream_error",
        message="Ollama returned an unsuccessful response.",
        details={
            "status_code": exc.response.status_code,
            "url": str(exc.request.url),
        },
    )


@app.exception_handler(httpx.HTTPError)
async def upstream_exception_handler(request: Request, exc: httpx.HTTPError):
    return error_response(
        status_code=503,
        code="upstream_unavailable",
        message="Ollama is unavailable.",
        details={"error": str(exc)},
    )


@app.get("/")
def root():
    return with_metadata(
        {
            "service": "atlas-api",
            "version": APP_VERSION,
            "status": "online",
            "endpoints": [
                "/",
                "/health",
                "/version",
                "/models",
                "/chat",
            ],
        }
    )


@app.get("/health")
def health():
    return with_metadata(
        {
            "status": "ok",
            "service": "atlas-api",
            "version": APP_VERSION,
            "ollama_base_url": OLLAMA_BASE_URL,
            "default_model": DEFAULT_MODEL,
        }
    )


@app.get("/version")
def version():
    return with_metadata(
        {
            "service": "atlas-api",
            "version": APP_VERSION,
        }
    )


@app.get("/models")
async def models():
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
        response.raise_for_status()
        data = response.json()

    return with_metadata(
        {
            "default_model": DEFAULT_MODEL,
            "models": data.get("models", []),
        }
    )


@app.post("/chat")
async def chat(request: ChatRequest):
    model = request.model or DEFAULT_MODEL

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": request.prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    return with_metadata(
        {
            "model": model,
            "response": data.get("response", ""),
            "done": data.get("done", False),
        }
    )
