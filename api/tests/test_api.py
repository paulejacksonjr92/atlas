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
                json={"response": "Apple TVs are assigned to VLAN 40 [1].", "done": True},
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
                                "id": "chunk-1",
                                "score": 0.88,
                                "payload": {
                                    "document_id": "document-1",
                                    "title": "Los Padrinos Network Notes",
                                    "text": "Apple TVs are assigned to VLAN 40.",
                                    "source": "manual-document",
                                    "metadata": {"client": "Los Padrinos"},
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
                                "text": "Apple TVs are on VLAN 40.",
                                "source": "recon",
                                "metadata": {"client": "Los Padrinos"},
                                "created_at": "2026-07-03T00:00:00+00:00",
                                "embedding_model": "nomic-embed-text",
                            },
                        }
                    ]
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
    assert payload["version"] == "0.6.0"
    assert "/version" in payload["endpoints"]
    assert "/status" in payload["endpoints"]
    assert "/embeddings" in payload["endpoints"]
    assert "/memory/search" in payload["endpoints"]
    assert "/documents" in payload["endpoints"]
    assert "/documents/search" in payload["endpoints"]
    assert "/chat/grounded" in payload["endpoints"]
    assert "/v1/models" in payload["endpoints"]
    assert "/v1/chat/completions" in payload["endpoints"]


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.6.0"


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
    assert payload["response"] == "Apple TVs are assigned to VLAN 40 [1]."
    assert payload["done"] is True


def test_grounded_chat(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post(
        "/chat/grounded",
        json={"prompt": "Which VLAN are the Apple TVs assigned?", "retrieval_limit": 3},
    )

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["model"] == "llama3.1:8b"
    assert payload["embedding_model"] == "nomic-embed-text"
    assert payload["prompt"] == "Which VLAN are the Apple TVs assigned?"
    assert payload["response"] == "Apple TVs are assigned to VLAN 40 [1]."
    assert payload["done"] is True
    assert payload["grounded"] is True
    assert payload["sources"][0]["score"] >= payload["sources"][1]["score"]
    assert {source["type"] for source in payload["sources"]} == {"memory", "document"}


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
                {"role": "user", "content": "Which VLAN are the Apple TVs assigned?"},
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["object"] == "chat.completion"
    assert payload["model"] == "atlas-grounded"
    content = payload["choices"][0]["message"]["content"]
    assert "Apple TVs are assigned to VLAN 40 [1]." in content
    assert "Sources:" in content


def test_openai_chat_completions_rejects_stream():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "atlas-grounded",
            "stream": True,
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "streaming_not_supported"


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
    assert payload["choices"][0]["message"]["content"] == "Apple TVs are assigned to VLAN 40 [1]."


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
            "text": "Apple TVs are on VLAN 40.",
            "source": "recon",
            "metadata": {"client": "Los Padrinos"},
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

    response = client.get("/memory/search", params={"query": "apple tv vlan"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["query"] == "apple tv vlan"
    assert payload["collection"] == "atlas_memory"
    assert payload["results"] == [
        {
            "memory_id": "memory-1",
            "score": 0.91,
            "text": "Apple TVs are on VLAN 40.",
            "source": "recon",
            "metadata": {"client": "Los Padrinos"},
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
            "title": "Los Padrinos Network Notes",
            "text": "Apple TVs are assigned to VLAN 40. The MDF switch is a Cisco Catalyst.",
            "source": "manual-document",
            "metadata": {"client": "Los Padrinos"},
            "chunk_size": 200,
            "chunk_overlap": 0,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["title"] == "Los Padrinos Network Notes"
    assert payload["collection"] == "atlas_documents"
    assert payload["embedding_model"] == "nomic-embed-text"
    assert payload["chunk_count"] == 1
    assert payload["stored"] is True
    assert any(request[0] == "PUT" and request[1].endswith("/collections/atlas_documents") for request in FakeAsyncClient.requests)
    assert any(request[0] == "PUT" and request[1].endswith("/points") for request in FakeAsyncClient.requests)


def test_search_documents(monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/documents/search", params={"query": "apple tv vlan"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["query"] == "apple tv vlan"
    assert payload["collection"] == "atlas_documents"
    assert payload["results"] == [
        {
            "document_id": "document-1",
            "chunk_id": "chunk-1",
            "score": 0.88,
            "title": "Los Padrinos Network Notes",
            "text": "Apple TVs are assigned to VLAN 40.",
            "source": "manual-document",
            "metadata": {"client": "Los Padrinos"},
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
