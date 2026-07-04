import asyncio
import json
from datetime import datetime, timezone
from uuid import uuid4

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.settings import (
    APP_VERSION,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_MODEL,
    DOCUMENT_COLLECTION,
    MEMORY_COLLECTION,
    OLLAMA_BASE_URL,
    OPENAI_COMPAT_MODEL,
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


class GroundedChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model: str | None = None
    retrieval_limit: int = Field(default=4, ge=1, le=10)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)


class OpenAIChatMessage(BaseModel):
    role: str
    content: str | None = ""


class OpenAIChatCompletionRequest(BaseModel):
    model: str = OPENAI_COMPAT_MODEL
    messages: list[OpenAIChatMessage] = Field(..., min_length=1)
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None


class EmbeddingRequest(BaseModel):
    text: str = Field(..., min_length=1)
    model: str | None = None


class MemoryWriteRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str | None = None
    metadata: dict = Field(default_factory=dict)
    collection: str | None = None
    model: str | None = None


class DocumentIngestRequest(BaseModel):
    title: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    source: str | None = None
    metadata: dict = Field(default_factory=dict)
    collection: str | None = None
    model: str | None = None
    chunk_size: int = Field(default=1200, ge=200, le=4000)
    chunk_overlap: int = Field(default=150, ge=0, le=1000)


UNSAFE_SOURCE_PATTERNS = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "secret",
    "secrets",
    "password",
    "token",
    "credential",
    "credentials",
    "backup",
    "backups",
    "log",
    "logs",
    ".sql",
    ".sqlite",
    ".db",
    ".dump",
]

UNSAFE_TEXT_PATTERNS = [
    "api_key",
    "apikey",
    "secret_key",
    "access_token",
    "refresh_token",
    "private_key",
    "smtp_password",
    "postgres_password",
    "database_url",
    "session_token",
]

KNOWLEDGE_POLICY = {
    "allowed": [
        "sanitized overviews",
        "reviewed architecture notes",
        "reviewed domain models",
        "boundary and ownership policies",
        "non-sensitive operational runbooks",
    ],
    "blocked": [
        ".env and local environment files",
        "passwords, tokens, API keys, and raw secrets",
        "database dumps and app databases",
        "backups and logs",
        "raw client, vendor, accounting, or production records",
        "unreviewed files marked unsafe",
    ],
    "required_metadata": {
        "project": "Owning system or app, such as Atlas, PatchCraft, or StudioServices.",
        "safety": "Expected values include sanitized, reviewed, policy, or public.",
        "type": "Source category, such as overview, architecture, domain-model, or policy.",
    },
}


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


def unix_timestamp() -> int:
    return int(datetime.now(timezone.utc).timestamp())


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


def chunk_text(text: str, chunk_size: int = 1200, chunk_overlap: int = 150) -> list[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks = []
    start = 0
    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))
        if end < len(clean_text):
            boundary = clean_text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(clean_text):
            break
        start = max(0, end - chunk_overlap)
    return chunks


def normalize_for_policy(value: str | None) -> str:
    return (value or "").replace("\\", "/").lower()


def knowledge_policy_violations(request: DocumentIngestRequest) -> list[str]:
    violations = []
    source = normalize_for_policy(request.source)
    title = normalize_for_policy(request.title)
    metadata = {str(key).lower(): value for key, value in request.metadata.items()}
    safety = str(metadata.get("safety", "")).lower()

    if safety in {"unsafe", "secret", "raw", "production-data"}:
        violations.append(f"metadata.safety={safety} is blocked")

    for pattern in UNSAFE_SOURCE_PATTERNS:
        if pattern in source or pattern in title:
            violations.append(f"source or title contains blocked pattern: {pattern}")

    searchable_text = normalize_for_policy(request.text[:5000])
    for pattern in UNSAFE_TEXT_PATTERNS:
        if pattern in searchable_text:
            violations.append(f"text contains blocked secret-shaped pattern: {pattern}")

    return sorted(set(violations))


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


async def generate_text(prompt: str, model: str | None = None) -> tuple[str, dict]:
    generation_model = model or DEFAULT_MODEL

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": generation_model,
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

    return generation_model, data


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


async def search_collection(collection: str, query: str, limit: int, model: str | None = None) -> tuple[str, list[dict]]:
    embedding_model, embedding = await create_embedding(query, model)

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{QDRANT_BASE_URL}/collections/{collection}/points/search",
            json={
                "vector": embedding,
                "limit": limit,
                "with_payload": True,
            },
        )
        if response.status_code == 404:
            return embedding_model, []
        response.raise_for_status()
        data = response.json()

    return embedding_model, data.get("result", [])


async def scroll_collection(collection: str, limit: int = 100) -> list[dict]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            f"{QDRANT_BASE_URL}/collections/{collection}/points/scroll",
            json={
                "limit": limit,
                "with_payload": True,
                "with_vector": False,
            },
        )
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()

    result = data.get("result", {})
    if isinstance(result, dict):
        return result.get("points", [])
    return result or []


def source_registry_entry(point: dict) -> dict:
    payload = point.get("payload", {})
    metadata = payload.get("metadata", {}) or {}
    return {
        "document_id": payload.get("document_id"),
        "title": payload.get("title"),
        "source": payload.get("source"),
        "project": metadata.get("project"),
        "safety": metadata.get("safety"),
        "type": metadata.get("type"),
        "chunk_count": payload.get("chunk_count"),
        "created_at": payload.get("created_at"),
        "embedding_model": payload.get("embedding_model"),
    }


def summarize_sources(points: list[dict]) -> list[dict]:
    sources = {}
    for point in points:
        entry = source_registry_entry(point)
        key = entry["document_id"] or f"{entry['title']}:{entry['source']}"
        if key not in sources:
            sources[key] = {**entry, "chunks_seen": 0}
        sources[key]["chunks_seen"] += 1
        sources[key]["chunk_count"] = max(
            sources[key].get("chunk_count") or 0,
            entry.get("chunk_count") or 0,
        )

    return sorted(
        sources.values(),
        key=lambda item: (item.get("project") or "", item.get("title") or ""),
    )


def memory_source(item: dict) -> dict:
    payload = item.get("payload", {})
    return {
        "type": "memory",
        "id": item.get("id"),
        "score": item.get("score"),
        "text": payload.get("text"),
        "source": payload.get("source"),
        "metadata": payload.get("metadata", {}),
        "created_at": payload.get("created_at"),
    }


def document_source(item: dict) -> dict:
    payload = item.get("payload", {})
    return {
        "type": "document",
        "id": item.get("id"),
        "document_id": payload.get("document_id"),
        "score": item.get("score"),
        "title": payload.get("title"),
        "text": payload.get("text"),
        "source": payload.get("source"),
        "metadata": payload.get("metadata", {}),
        "chunk_index": payload.get("chunk_index"),
        "chunk_count": payload.get("chunk_count"),
        "created_at": payload.get("created_at"),
    }


async def retrieve_context(query: str, limit: int, min_score: float) -> tuple[str, list[dict]]:
    memory_limit = max(1, limit)
    document_limit = max(1, limit)
    memory_result, document_result = await asyncio.gather(
        search_collection(MEMORY_COLLECTION, query, memory_limit),
        search_collection(DOCUMENT_COLLECTION, query, document_limit),
    )

    embedding_model = document_result[0]
    sources = [memory_source(item) for item in memory_result[1]]
    sources.extend(document_source(item) for item in document_result[1])
    sources = [source for source in sources if source.get("score") is None or source["score"] >= min_score]
    sources.sort(key=lambda source: source.get("score") or 0, reverse=True)
    return embedding_model, sources[:limit]


def build_grounded_prompt(question: str, sources: list[dict]) -> str:
    if not sources:
        context = "No relevant Atlas memory or document context was found."
    else:
        context_blocks = []
        for index, source in enumerate(sources, start=1):
            label = f"[{index}] {source['type']}"
            if source.get("title"):
                label += f" - {source['title']}"
            if source.get("source"):
                label += f" ({source['source']})"
            context_blocks.append(f"{label}\n{source.get('text') or ''}")
        context = "\n\n".join(context_blocks)

    return f"""You are Atlas, the AI operating system for technical operations.
Answer only from the provided Atlas context. If the context does not contain the answer, say you do not know from Atlas memory yet.
Be concise, technical, and cite sources using bracket numbers like [1].

Atlas context:
{context}

Question:
{question}

Answer:"""


def latest_user_message(messages: list[OpenAIChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user" and message.content:
            return message.content
    for message in reversed(messages):
        if message.content:
            return message.content
    return ""


def openai_chat_response(model: str, content: str) -> dict:
    completion_id = f"chatcmpl-{request_id()}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": unix_timestamp(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def openai_chat_stream(model: str, content: str) -> StreamingResponse:
    completion_id = f"chatcmpl-{request_id()}"
    created = unix_timestamp()

    async def events():
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": content,
                    },
                    "finish_reason": None,
                }
            ],
        }
        done = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


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
                "/chat/grounded",
                "/v1/models",
                "/v1/chat/completions",
                "/embeddings",
                "/memory",
                "/memory/search",
                "/documents",
                "/documents/search",
                "/knowledge/policy",
                "/knowledge/sources",
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
            "document_collection": DOCUMENT_COLLECTION,
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


@app.get("/v1/models")
def openai_models():
    created = unix_timestamp()
    return {
        "object": "list",
        "data": [
            {
                "id": OPENAI_COMPAT_MODEL,
                "object": "model",
                "created": created,
                "owned_by": SERVICE_NAME,
            }
        ],
    }


@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatCompletionRequest):
    prompt = latest_user_message(request.messages)
    if not prompt:
        return error_response(
            status_code=422,
            code="missing_prompt",
            message="No user message content was provided.",
            details=None,
        )

    if request.model == OPENAI_COMPAT_MODEL:
        embedding_model, sources = await retrieve_context(prompt, limit=4, min_score=0.2)
        grounded_prompt = build_grounded_prompt(prompt, sources)
        model, data = await generate_text(grounded_prompt, DEFAULT_MODEL)
        content = data.get("response", "")
        if sources:
            source_lines = []
            for index, source in enumerate(sources, start=1):
                label = source.get("title") or source.get("source") or source.get("type")
                source_lines.append(f"[{index}] {label}")
            content = f"{content}\n\nSources:\n" + "\n".join(source_lines)
        if request.stream:
            return openai_chat_stream(request.model, content)
        return openai_chat_response(request.model, content)

    model, data = await generate_text(prompt, request.model)
    content = data.get("response", "")
    if request.stream:
        return openai_chat_stream(model, content)
    return openai_chat_response(model, content)


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
    model, data = await generate_text(request.prompt, request.model)

    return with_metadata(
        {
            "model": model,
            "response": data.get("response", ""),
            "done": data.get("done", False),
        }
    )


@app.post("/chat/grounded")
async def grounded_chat(request: GroundedChatRequest):
    embedding_model, sources = await retrieve_context(
        request.prompt,
        request.retrieval_limit,
        request.min_score,
    )
    grounded_prompt = build_grounded_prompt(request.prompt, sources)
    model, data = await generate_text(grounded_prompt, request.model)

    return with_metadata(
        {
            "model": model,
            "embedding_model": embedding_model,
            "prompt": request.prompt,
            "response": data.get("response", ""),
            "done": data.get("done", False),
            "grounded": bool(sources),
            "sources": sources,
        }
    )


@app.get("/knowledge/policy")
def knowledge_policy():
    return with_metadata(KNOWLEDGE_POLICY)


@app.get("/knowledge/sources")
async def knowledge_sources(
    collection: str | None = None,
    limit: int = Query(100, ge=1, le=1000),
):
    document_collection = collection or DOCUMENT_COLLECTION
    points = await scroll_collection(document_collection, limit)
    sources = summarize_sources(points)

    return with_metadata(
        {
            "collection": document_collection,
            "source_count": len(sources),
            "sources": sources,
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


@app.post("/documents")
async def ingest_document(request: DocumentIngestRequest):
    violations = knowledge_policy_violations(request)
    if violations:
        return error_response(
            status_code=422,
            code="knowledge_policy_violation",
            message="Document was blocked by Atlas knowledge ingestion policy.",
            details={"violations": violations, "policy_endpoint": "/knowledge/policy"},
        )

    collection = request.collection or DOCUMENT_COLLECTION
    document_id = request_id()
    ingested_at = utc_now()
    chunks = chunk_text(request.text, request.chunk_size, request.chunk_overlap)
    if not chunks:
        return error_response(
            status_code=422,
            code="empty_document",
            message="Document text did not contain ingestible content.",
            details=None,
        )

    points = []
    embedding_model = request.model or DEFAULT_EMBEDDING_MODEL
    vector_size = None

    for index, chunk in enumerate(chunks):
        embedding_model, embedding = await create_embedding(chunk, request.model)
        vector_size = len(embedding)
        points.append(
            {
                "id": request_id(),
                "vector": embedding,
                "payload": {
                    "document_id": document_id,
                    "title": request.title,
                    "text": chunk,
                    "source": request.source,
                    "metadata": request.metadata,
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                    "created_at": ingested_at,
                    "embedding_model": embedding_model,
                },
            }
        )

    async with httpx.AsyncClient(timeout=60.0) as client:
        await ensure_memory_collection(client, collection, vector_size or 0)
        response = await client.put(
            f"{QDRANT_BASE_URL}/collections/{collection}/points",
            params={"wait": "true"},
            json={"points": points},
        )
        response.raise_for_status()

    return with_metadata(
        {
            "document_id": document_id,
            "title": request.title,
            "collection": collection,
            "source": request.source,
            "embedding_model": embedding_model,
            "chunk_count": len(chunks),
            "stored": True,
        }
    )


@app.get("/documents/search")
async def search_documents(
    query: str = Query(..., min_length=1),
    limit: int = Query(5, ge=1, le=25),
    collection: str | None = None,
    model: str | None = None,
):
    document_collection = collection or DOCUMENT_COLLECTION
    embedding_model, found = await search_collection(document_collection, query, limit, model)

    results = []
    for item in found:
        payload = item.get("payload", {})
        results.append(
            {
                "document_id": payload.get("document_id"),
                "chunk_id": item.get("id"),
                "score": item.get("score"),
                "title": payload.get("title"),
                "text": payload.get("text"),
                "source": payload.get("source"),
                "metadata": payload.get("metadata", {}),
                "chunk_index": payload.get("chunk_index"),
                "chunk_count": payload.get("chunk_count"),
                "created_at": payload.get("created_at"),
                "embedding_model": payload.get("embedding_model"),
            }
        )

    return with_metadata(
        {
            "query": query,
            "collection": document_collection,
            "embedding_model": embedding_model,
            "results": results,
        }
    )
