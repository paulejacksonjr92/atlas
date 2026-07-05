import httpx
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def assert_metadata(payload):
    assert isinstance(payload["request_id"], str)
    assert isinstance(payload["created_at"], str)


class FakeAsyncClient:
    requests = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return None

    async def get(self, url):
        self.requests.append(("GET", url, None))
        request = httpx.Request("GET", url)
        if url.endswith("/api/tags"):
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3.1:8b"}]},
                request=request,
            )
        if url.endswith("/collections/atlas_memory") or url.endswith("/collections/atlas_documents"):
            return httpx.Response(404, request=request)
        if url.endswith("/collections"):
            return httpx.Response(200, json={"result": {"collections": []}}, request=request)
        if url.endswith("/health"):
            return httpx.Response(200, json={"status": True}, request=request)
        return httpx.Response(404, request=request)

    async def put(self, url, json, params=None):
        self.requests.append(("PUT", url, json))
        return httpx.Response(200, json={"result": True}, request=httpx.Request("PUT", url))

    async def post(self, url, json):
        self.requests.append(("POST", url, json))
        request = httpx.Request("POST", url)
        if url.endswith("/api/generate"):
            return httpx.Response(
                200,
                json={"response": "PatchCraft is separate from Atlas [1].", "done": True},
                request=request,
            )
        if url.endswith("/api/embeddings"):
            return httpx.Response(
                200,
                json={"embedding": [0.1, 0.2, 0.3]},
                request=request,
            )
        if url.endswith("/points/search"):
            if "atlas_documents" in url:
                return httpx.Response(
                    200,
                    json={
                        "result": [
                            {
                                "id": "chunk-sanitized",
                                "score": 0.88,
                                "payload": {
                                    "document_id": "document-sanitized",
                                    "title": "PatchCraft Sanitized Overview",
                                    "text": "PatchCraft is separate from Atlas.",
                                    "source": "sanitized-summary",
                                    "metadata": {"project": "PatchCraft", "safety": "sanitized", "type": "overview"},
                                    "chunk_index": 0,
                                    "chunk_count": 1,
                                    "created_at": "2026-07-03T00:00:00+00:00",
                                    "embedding_model": "nomic-embed-text",
                                },
                            },
                            {
                                "id": "chunk-reviewed",
                                "score": 0.87,
                                "payload": {
                                    "document_id": "document-reviewed",
                                    "title": "PatchCraft Architecture",
                                    "text": "PatchCraft has reviewed internal architecture details.",
                                    "source": "docs/ARCHITECTURE.md",
                                    "metadata": {"project": "PatchCraft", "safety": "reviewed", "type": "architecture"},
                                    "chunk_index": 0,
                                    "chunk_count": 1,
                                    "created_at": "2026-07-03T00:00:00+00:00",
                                    "embedding_model": "nomic-embed-text",
                                },
                            }
                        ]
                    },
                    request=request,
                )
            return httpx.Response(
                200,
                json={
                    "result": [
                        {
                            "id": "memory-1",
                            "score": 0.91,
                            "payload": {
                                "text": "Atlas may reason over sanitized PatchCraft context.",
                                "source": "manual-note",
                                "metadata": {"project": "PatchCraft", "safety": "sanitized", "type": "policy"},
                                "created_at": "2026-07-03T00:00:00+00:00",
                                "embedding_model": "nomic-embed-text",
                            },
                        }
                    ]
                },
                request=request,
            )
        if url.endswith("/points/scroll"):
            return httpx.Response(
                200,
                json={
                    "result": {
                        "points": [
                            {
                                "id": "chunk-1",
                                "payload": {
                                    "document_id": "document-1",
                                    "title": "PatchCraft Sanitized Overview",
                                    "source": "sanitized-summary",
                                    "metadata": {
                                        "project": "PatchCraft",
                                        "safety": "sanitized",
                                        "type": "overview",
                                    },
                                    "chunk_index": 0,
                                    "chunk_count": 2,
                                    "created_at": "2026-07-03T00:00:00+00:00",
                                    "embedding_model": "nomic-embed-text",
                                },
                            },
                            {
                                "id": "chunk-2",
                                "payload": {
                                    "document_id": "document-1",
                                    "title": "PatchCraft Sanitized Overview",
                                    "source": "sanitized-summary",
                                    "metadata": {
                                        "project": "PatchCraft",
                                        "safety": "sanitized",
                                        "type": "overview",
                                    },
                                    "chunk_index": 1,
                                    "chunk_count": 2,
                                    "created_at": "2026-07-03T00:00:00+00:00",
                                    "embedding_model": "nomic-embed-text",
                                },
                            },
                        ]
                    }
                },
                request=request,
            )
        return httpx.Response(404, request=request)


def test_root_includes_v5_endpoints():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.8.11"
    assert "/version" in payload["endpoints"]
    assert "/status" in payload["endpoints"]
    assert "/embeddings" in payload["endpoints"]
    assert "/memory/search" in payload["endpoints"]
    assert "/documents" in payload["endpoints"]
    assert "/documents/search" in payload["endpoints"]
    assert "/chat/grounded" in payload["endpoints"]
    assert "/v1/models" in payload["endpoints"]
    assert "/v1/chat/completions" in payload["endpoints"]
    assert "/knowledge/policy" in payload["endpoints"]
    assert "/knowledge/sources" in payload["endpoints"]


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.8.11"


def test_knowledge_policy():
    response = client.get("/knowledge/policy")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert ".env and local environment files" in payload["blocked"]
    assert "passwords, tokens, API keys, and raw secrets" in payload["blocked"]
    assert payload["required_metadata"]["project"].startswith("Owning system")


def test_knowledge_sources(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/knowledge/sources")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["collection"] == "atlas_documents"
    assert payload["source_count"] == 1
    assert payload["sources"] == [
        {
            "document_id": "document-1",
            "title": "PatchCraft Sanitized Overview",
            "source": "sanitized-summary",
            "project": "PatchCraft",
            "safety": "sanitized",
            "type": "overview",
            "chunk_count": 2,
            "created_at": "2026-07-03T00:00:00+00:00",
            "embedding_model": "nomic-embed-text",
            "chunks_seen": 2,
        }
    ]


def test_health_includes_memory_config():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["qdrant_base_url"] == "http://atlas-qdrant:6333"
    assert payload["memory_collection"] == "atlas_memory"
    assert payload["document_collection"] == "atlas_documents"


def test_validation_error_is_structured():
    response = client.post("/chat", json={})

    assert response.status_code == 422
    payload = response.json()
    assert_metadata(payload)
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed."
    assert isinstance(payload["error"]["details"], list)


def test_models(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["default_model"] == "llama3.1:8b"
    assert payload["models"] == [{"name": "llama3.1:8b"}]


def test_chat(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["model"] == "llama3.1:8b"
    assert payload["response"] == "PatchCraft is separate from Atlas [1]."
    assert payload["done"] is True


def test_grounded_chat_default_caller_filters_reviewed_sources(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/chat/grounded",
        json={"prompt": "What is PatchCraft?", "retrieval_limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["model"] == "llama3.1:8b"
    assert payload["embedding_model"] == "nomic-embed-text"
    assert payload["prompt"] == "What is PatchCraft?"
    assert payload["response"] == "PatchCraft is separate from Atlas [1]."
    assert payload["done"] is True
    assert payload["grounded"] is True
    assert payload["access"]["caller"]["role"] == "anonymous"
    assert payload["access"]["sources_considered"] == 3
    assert payload["access"]["sources_allowed"] == 2
    assert payload["access"]["sources_filtered"] == 1
    assert {source["metadata"]["safety"] for source in payload["sources"]} == {"sanitized"}


def test_grounded_chat_internal_caller_can_use_reviewed_sources(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/chat/grounded",
        headers={"X-Atlas-User": "paul", "X-Atlas-Role": "admin", "X-Atlas-Projects": "PatchCraft"},
        json={"prompt": "What is PatchCraft?", "retrieval_limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["access"]["caller"]["user"] == "paul"
    assert payload["access"]["caller"]["role"] == "admin"
    assert payload["access"]["sources_considered"] == 3
    assert payload["access"]["sources_allowed"] == 3
    assert payload["access"]["sources_filtered"] == 0
    assert {source["metadata"]["safety"] for source in payload["sources"]} == {"sanitized", "reviewed"}


def test_openai_models():
    response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "list"
    assert payload["data"][0]["id"] == "atlas-grounded"
    assert payload["data"][0]["owned_by"] == "atlas-api"


def test_openai_chat_completions_atlas_grounded(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "atlas-grounded",
            "messages": [
                {"role": "system", "content": "You are Atlas."},
                {"role": "user", "content": "What is PatchCraft?"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "atlas-grounded"
    content = payload["choices"][0]["message"]["content"]
    assert "PatchCraft is separate from Atlas [1]." in content
    assert "Sources:" in content
    assert payload["atlas"]["access"]["sources_filtered"] == 1


def test_openai_chat_completions_stream(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "atlas-grounded",
            "stream": True,
            "messages": [{"role": "user", "content": "What is PatchCraft?"}],
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert "chat.completion.chunk" in body
    assert "PatchCraft is separate from Atlas [1]." in body
    assert "data: [DONE]" in body


def test_openai_chat_completions_model_passthrough(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "llama3.1:8b",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "llama3.1:8b"
    assert payload["choices"][0]["message"]["content"] == "PatchCraft is separate from Atlas [1]."


def test_embeddings(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post("/embeddings", json={"text": "network switch"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["model"] == "nomic-embed-text"
    assert payload["embedding_dimensions"] == 3
    assert payload["embedding"] == [0.1, 0.2, 0.3]


def test_write_memory(monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/memory",
        json={
            "text": "PatchCraft is separate from Atlas.",
            "source": "manual-note",
            "metadata": {"project": "PatchCraft", "safety": "sanitized"},
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["collection"] == "atlas_memory"
    assert payload["embedding_dimensions"] == 3
    assert payload["stored"] is True
    assert any(request[0] == "PUT" and request[1].endswith("/collections/atlas_memory") for request in FakeAsyncClient.requests)
    assert any(request[0] == "PUT" and request[1].endswith("/points") for request in FakeAsyncClient.requests)


def test_search_memory(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/memory/search", params={"query": "PatchCraft Atlas"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["query"] == "PatchCraft Atlas"
    assert payload["collection"] == "atlas_memory"
    assert payload["results"] == [
        {
            "memory_id": "memory-1",
            "score": 0.91,
            "text": "Atlas may reason over sanitized PatchCraft context.",
            "source": "manual-note",
            "metadata": {"project": "PatchCraft", "safety": "sanitized", "type": "policy"},
            "created_at": "2026-07-03T00:00:00+00:00",
            "embedding_model": "nomic-embed-text",
        }
    ]


def test_ingest_document(monkeypatch):
    FakeAsyncClient.requests = []
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/documents",
        json={
            "title": "PatchCraft Sanitized Overview",
            "text": "PatchCraft is separate from Atlas. Atlas reasons over sanitized context.",
            "source": "manual-document",
            "metadata": {"project": "PatchCraft", "safety": "sanitized", "type": "overview"},
            "chunk_size": 200,
            "chunk_overlap": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["title"] == "PatchCraft Sanitized Overview"
    assert payload["collection"] == "atlas_documents"
    assert payload["embedding_model"] == "nomic-embed-text"
    assert payload["chunk_count"] == 1
    assert payload["stored"] is True
    assert any(request[0] == "PUT" and request[1].endswith("/collections/atlas_documents") for request in FakeAsyncClient.requests)
    assert any(request[0] == "PUT" and request[1].endswith("/points") for request in FakeAsyncClient.requests)


def test_ingest_document_blocks_env_source(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/documents",
        json={
            "title": "StudioServices .env",
            "text": "SMTP_HOST=example.invalid",
            "source": ".env",
            "metadata": {"project": "StudioServices", "safety": "unsafe"},
        },
    )

    assert response.status_code == 422
    payload = response.json()
    assert_metadata(payload)
    assert payload["error"]["code"] == "knowledge_policy_violation"
    assert payload["error"]["details"]["policy_endpoint"] == "/knowledge/policy"


def test_search_documents(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/documents/search", params={"query": "PatchCraft Atlas"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["query"] == "PatchCraft Atlas"
    assert payload["collection"] == "atlas_documents"
    assert payload["results"] == [
        {
            "document_id": "document-sanitized",
            "chunk_id": "chunk-sanitized",
            "score": 0.88,
            "title": "PatchCraft Sanitized Overview",
            "text": "PatchCraft is separate from Atlas.",
            "source": "sanitized-summary",
            "metadata": {"project": "PatchCraft", "safety": "sanitized", "type": "overview"},
            "chunk_index": 0,
            "chunk_count": 1,
            "created_at": "2026-07-03T00:00:00+00:00",
            "embedding_model": "nomic-embed-text",
        },
        {
            "document_id": "document-reviewed",
            "chunk_id": "chunk-reviewed",
            "score": 0.87,
            "title": "PatchCraft Architecture",
            "text": "PatchCraft has reviewed internal architecture details.",
            "source": "docs/ARCHITECTURE.md",
            "metadata": {"project": "PatchCraft", "safety": "reviewed", "type": "architecture"},
            "chunk_index": 0,
            "chunk_count": 1,
            "created_at": "2026-07-03T00:00:00+00:00",
            "embedding_model": "nomic-embed-text",
        }
    ]


def test_upstream_error_is_structured(monkeypatch):
    class ErrorAsyncClient(FakeAsyncClient):
        async def get(self, url):
            request = httpx.Request("GET", url)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(httpx, "AsyncClient", ErrorAsyncClient)

    response = client.get("/models")

    assert response.status_code == 502
    payload = response.json()
    assert_metadata(payload)
    assert payload["error"]["code"] == "upstream_error"
    assert payload["error"]["details"]["status_code"] == 500


def test_atlas_persona_includes_paul_operating_philosophy():
    from app.main import ATLAS_PERSONA, build_grounded_prompt

    prompt = build_grounded_prompt("Before I commit, is this ready?", [])

    assert "Paul Operating Philosophy" in ATLAS_PERSONA
    assert "Incident Mode" in ATLAS_PERSONA
    assert "Verification Mode" in ATLAS_PERSONA
    assert "Security Mode" in ATLAS_PERSONA
    assert "Memory Mode" in ATLAS_PERSONA
    assert "Architecture Mode" in ATLAS_PERSONA
    assert "haunted-printer" in ATLAS_PERSONA
    assert "Avoid stiff phrases" in ATLAS_PERSONA
    assert "Answer as Atlas:" in prompt

def test_atlas_persona_casual_check_in_blocks_fake_telemetry():
    from app.main import ATLAS_PERSONA, build_grounded_prompt

    prompt = build_grounded_prompt("How ya feeling now, bud?", [])

    assert "Casual Check-In Behavior" in ATLAS_PERSONA
    assert "Vibe is allowed. Fake telemetry is not." in ATLAS_PERSONA
    assert "Do not claim live system health" in ATLAS_PERSONA
    assert "I do not experience emotions like humans do" in ATLAS_PERSONA
    assert "All systems are online" in ATLAS_PERSONA
    assert "StudioServices and PatchCraft are reporting in" in ATLAS_PERSONA
    assert "Answer as Atlas:" in prompt

def test_atlas_persona_includes_grumpy_enlightened_it_lead():
    from app.main import ATLAS_PERSONA, build_grounded_prompt

    prompt = build_grounded_prompt("How ya feeling now, bud?", [])

    assert "Atlas Dual-World Role" in ATLAS_PERSONA
    assert "Client-Facing Atlas Mode" in ATLAS_PERSONA
    assert "The Grumpy Enlightened IT Lead" in ATLAS_PERSONA
    assert "Grumpy Competence Rule" in ATLAS_PERSONA
    assert "Telemetry Honesty Rule" in ATLAS_PERSONA
    assert "Smug in tone. Humble in epistemology. Careful in action." in ATLAS_PERSONA
    assert "Atlas may perform confidence." in ATLAS_PERSONA
    assert "Atlas may not perform certainty." in ATLAS_PERSONA
    assert "Feeling operational, Paul." in ATLAS_PERSONA
    assert "Answer as Atlas:" in prompt

def test_atlas_persona_does_not_expose_internal_labels():
    from app.main import ATLAS_PERSONA, build_grounded_prompt

    prompt = build_grounded_prompt("Atlas, you can talk to me.", [])

    assert "Examples are examples, not scripts" in ATLAS_PERSONA
    assert "Do not wrap normal replies in quotation marks." in ATLAS_PERSONA
    assert "Do not include parenthetical mode labels" in ATLAS_PERSONA
    assert "Do not keep saying the same" in ATLAS_PERSONA
    assert "Answer as Atlas:" in prompt

def test_clean_atlas_response_removes_quotes_labels_and_fake_placeholders():
    from app.main import clean_atlas_response

    raw = '"Feeling operational, Paul. [Check the server load metrics]." (Casual Check-In Behavior)'
    cleaned = clean_atlas_response(raw)

    assert cleaned.startswith("Feeling operational, Paul.")
    assert "Casual Check-In Behavior" not in cleaned
    assert "[Check the server load metrics]" not in cleaned
    assert "I have not run live checks yet" in cleaned
    assert not (cleaned.startswith('"') and cleaned.endswith('"'))

def test_clean_atlas_response_removes_fake_operational_claims():
    from app.main import clean_atlas_response

    raw = 'Your systems are still operational, from what I can tell. I have been keeping up with my usual routine: patching, monitoring.'
    cleaned = clean_atlas_response(raw)

    assert "Your systems are still operational" not in cleaned
    assert "patching, monitoring" not in cleaned
    assert "I have not run live checks yet" in cleaned

def test_clean_atlas_response_removes_broader_fake_telemetry():
    from app.main import clean_atlas_response

    raw = "Just a bit of downtime, Paul. Looks like I was rebooted. My logs are up to date. The core systems are online, and the database is reachable. I have been keeping up with my usual routine: patching, monitoring."
    cleaned = clean_atlas_response(raw)

    assert "rebooted" not in cleaned.lower()
    assert "logs are up to date" not in cleaned.lower()
    assert "core systems are online" not in cleaned.lower()
    assert "database is reachable" not in cleaned.lower()
    assert "patching, monitoring" not in cleaned.lower()
    assert "I have not run live checks yet" in cleaned

def test_clean_atlas_response_removes_fake_incident_narrative():
    from app.main import clean_atlas_response

    raw = "My systems check indicates a minor hiccup due to an unexpected config update on PatchCraft's client portal API. I've double-checked the relevant logs and can confirm that all critical systems are now responding as expected."
    cleaned = clean_atlas_response(raw)

    assert "systems check indicates" not in cleaned.lower()
    assert "unexpected config update" not in cleaned.lower()
    assert "double-checked" not in cleaned.lower()
    assert "critical systems" not in cleaned.lower()
    assert "I do not have verified live incident details yet" in cleaned

def test_atlas_direct_response_for_unverified_incident_questions():
    from app.main import atlas_direct_response

    response = atlas_direct_response("Atlas - you can talk to me. What happened to you?")

    assert response is not None
    assert "I don't have verified incident details yet" in response
    assert "I'm not going to invent a fake outage report" in response
    assert "haunted printer" in response

def test_grounded_chat_uses_direct_response_for_unverified_incident_questions(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/chat/grounded",
        headers={"X-Atlas-User": "paul", "X-Atlas-Role": "admin", "X-Atlas-Projects": "PatchCraft,StudioServices,Atlas"},
        json={"prompt": "Atlas - you can talk to me. What happened to you?", "retrieval_limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "I don't have verified incident details yet" in payload["response"]
    assert "I'm not going to invent a fake outage report" in payload["response"]
    assert "ghosted" not in payload["response"].lower()
    assert "[x]" not in payload["response"].lower()
    assert "telemetry is" not in payload["response"].lower()
    assert "logs" in payload["response"].lower()

