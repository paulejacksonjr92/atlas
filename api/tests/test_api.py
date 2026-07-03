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
        if url.endswith("/collections/atlas_memory"):
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
                json={"response": "hi", "done": True},
                request=request,
            )
        if url.endswith("/api/embeddings"):
            return httpx.Response(
                200,
                json={"embedding": [0.1, 0.2, 0.3]},
                request=request,
            )
        if url.endswith("/points/search"):
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


def test_root_includes_v3_endpoints():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.3.0"
    assert "/version" in payload["endpoints"]
    assert "/status" in payload["endpoints"]
    assert "/embeddings" in payload["endpoints"]
    assert "/memory/search" in payload["endpoints"]


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.3.0"


def test_health_includes_memory_config():
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["qdrant_base_url"] == "http://atlas-qdrant:6333"
    assert payload["memory_collection"] == "atlas_memory"


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
    assert payload["response"] == "hi"
    assert payload["done"] is True


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
