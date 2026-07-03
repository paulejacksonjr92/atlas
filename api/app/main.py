import asyncio
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.settings import (
    APP_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    MEMORY_COLLECTION,
    OLLAMA_BASE_URL,
    OPENWEBUI_BASE_URL,
    QDRANT_BASE_URL,
    REDIS_HOST,
    REDIS_PORT,
    SERVICE_NAME,
)

app = FastAPI(title="Atlas API", version=APP_VERSION)


class ChatRequest(BaseModel):
    prompt: str
    model: str | None = None


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str | None = None


class MemoryWriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str | None = None
    metadata: dict = Field(default_factory=dict)
    collection: str | None = None
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
        message="An upstream service returned an unsuccessful response.",
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
        message="An upstream service is unavailable.",
        details={"error": str(exc)},
    )


def parse_embedding(data: dict) -> list[float]:
    if "embedding" in data:
        return data["embedding"]
    if "embeddings" in data and data["embeddings"]:
        return data["embeddings"][0]
    return []


async def create_embedding(text: str, model: str | None = None) -> tuple[str, list[float]]:
    embedding_model = model or DEFAULT_EMBEDDING_MODEL

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={
                "model": embedding_model,
                "prompt": text,
            },
        )
        response.raise_for_status()
        data = response.json()

    return embedding_model, parse_embedding(data)


async def ensure_memory_collection(client: httpx.AsyncClient, collection: str, vector_size: int):
    response = await client.get(f"{QDRANT_BASE_URL}/collections/{collection}")
    if response.status_code == 200:
        return
    if response.status_code != 404:
        response.raise_for_status()

    create_response = await client.put(
        f"{QDRANT_BASE_URL}/collections/{collection}",
        json={
            "vectors": {
                "size": vector_size,
                "distance": "Cosine",
            }
        },
    )
    create_response.raise_for_status()


async def check_http_service(client: httpx.AsyncClient, name: str, url: str) -> dict:
    try:
        response = await client.get(url)
        return {
            "name": name,
            "status": "ok" if response.status_code < 500 else "error",
            "status_code": response.status_code,
        }
    except httpx.HTTPError as exc:
        return {
            "name": name,
            "status": "error",
            "error": str(exc),
        }


async def check_redis() -> dict:
    try:
        reader, writer = await asyncio.open_connection(REDIS_HOST, REDIS_PORT)
        writer.write(b"*1\r\n$4\r\nPING\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.readline(), timeout=5.0)
        writer.close()
        await writer.wait_closed()
        return {
            "name": "redis",
            "status": "ok" if response.startswith(b"+PONG") else "error",
        }
    except Exception as exc:
        return {
            "name": "redis",
            "status": "error",
            "error": str(exc),
        }


@app.get("/")
def root():
    return with_metadata(
        {
            "service": SERVICE_NAME,
            "version": APP_VERSION,
            "status": "online",
            "endpoints": [
                "/",
                "/health",
                "/version",
                "/status",
                "/models",
                "/chat",
                "/embeddings",
                "/memory",
                "/memory/search",
            ],
        }
    )


@app.get("/health")
def health():
    return with_metadata(
        {
            "status": "ok",
            "service": SERVICE_NAME,
            "version": APP_VERSION,
            "ollama_base_url": OLLAMA_BASE_URL,
            "qdrant_base_url": QDRANT_BASE_URL,
            "default_model": DEFAULT_MODEL,
            "default_embedding_model": DEFAULT_EMBEDDING_MODEL,
            "memory_collection": MEMORY_COLLECTION,
        }
    )


@app.get("/version")
def version():
    return with_metadata(
        {
            "service": SERVICE_NAME,
            "version": APP_VERSION,
        }
    )


@app.get("/status")
async def status():
    async with httpx.AsyncClient(timeout=10.0) as client:
        services = await asyncio.gather(
            check_http_service(client, "ollama", f"{OLLAMA_BASE_URL}/api/tags"),
            check_http_service(client, "qdrant", f"{QDRANT_BASE_URL}/collections"),
            check_http_service(client, "openwebui", f"{OPENWEBUI_BASE_URL}/health"),
            check_redis(),
        )

    state = "ok" if all(service["status"] == "ok" for service in services) else "degraded"
    return with_metadata(
        {
            "status": state,
            "services": services,
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


@app.post("/embeddings")
async def embeddings(request: EmbeddingRequest):
    model, embedding = await create_embedding(request.text, request.model)

    return with_metadata(
        {
            "model": model,
            "embedding_dimensions": len(embedding),
            "embedding": embedding,
        }
    )


@app.post("/memory")
async def write_memory(request: MemoryWriteRequest):
    collection = request.collection or MEMORY_COLLECTION
    model, embedding = await create_embedding(request.text, request.model)
    memory_id = request_id()
    captured_at = utc_now()
    payload = {
        "text": request.text,
        "source": request.source,
        "metadata": request.metadata,
        "created_at": captured_at,
        "embedding_model": model,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        await ensure_memory_collection(client, collection, len(embedding))
        response = await client.put(
            f"{QDRANT_BASE_URL}/collections/{collection}/points",
            params={"wait": "true"},
            json={
                "points": [
                    {
                        "id": memory_id,
                        "vector": embedding,
                        "payload": payload,
                    }
                ]
            },
        )
        response.raise_for_status()

    return with_metadata(
        {
            "memory_id": memory_id,
            "collection": collection,
            "embedding_model": model,
            "embedding_dimensions": len(embedding),
            "stored": True,
        }
    )


@app.get("/memory/search")
async def search_memory(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=25),
    collection: str | None = None,
    model: str | None = None,
):
    memory_collection = collection or MEMORY_COLLECTION
    embedding_model, embedding = await create_embedding(query, model)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{QDRANT_BASE_URL}/collections/{memory_collection}/points/search",
            json={
                "vector": embedding,
                "limit": limit,
                "with_payload": True,
            },
        )
        response.raise_for_status()
        data = response.json()

    results = []
    for item in data.get("result", []):
        payload = item.get("payload", {})
        results.append(
            {
                "memory_id": item.get("id"),
                "score": item.get("score"),
                "text": payload.get("text"),
                "source": payload.get("source"),
                "metadata": payload.get("metadata", {}),
                "created_at": payload.get("created_at"),
                "embedding_model": payload.get("embedding_model"),
            }
        )

    return with_metadata(
        {
            "query": query,
            "collection": memory_collection,
            "embedding_model": embedding_model,
            "results": results,
        }
    )
