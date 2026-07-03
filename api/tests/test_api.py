import httpx
from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def assert_metadata(payload):
    assert isinstance(payload["request_id"], str)
    assert isinstance(payload["created_at"], str)


def test_root_includes_metadata_and_version_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.2.0"
    assert "/version" in payload["endpoints"]


def test_version():
    response = client.get("/version")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["service"] == "atlas-api"
    assert payload["version"] == "0.2.0"


def test_validation_error_is_structured():
    response = client.post("/chat", json={})

    assert response.status_code == 422
    payload = response.json()
    assert_metadata(payload)
    assert payload["error"]["code"] == "validation_error"
    assert payload["error"]["message"] == "Request validation failed."
    assert isinstance(payload["error"]["details"], list)


def test_models(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            return httpx.Response(
                200,
                json={"models": [{"name": "llama3.1:8b"}]},
                request=httpx.Request("GET", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/models")

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["default_model"] == "llama3.1:8b"
    assert payload["models"] == [{"name": "llama3.1:8b"}]


def test_chat(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, json):
            assert json["model"] == "llama3.1:8b"
            assert json["prompt"] == "hello"
            assert json["stream"] is False
            return httpx.Response(
                200,
                json={"response": "hi", "done": True},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.post("/chat", json={"prompt": "hello"})

    assert response.status_code == 200
    payload = response.json()
    assert_metadata(payload)
    assert payload["model"] == "llama3.1:8b"
    assert payload["response"] == "hi"
    assert payload["done"] is True


def test_upstream_error_is_structured(monkeypatch):
    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url):
            request = httpx.Request("GET", url)
            response = httpx.Response(500, request=request)
            raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    response = client.get("/models")

    assert response.status_code == 502
    payload = response.json()
    assert_metadata(payload)
    assert payload["error"]["code"] == "upstream_error"
    assert payload["error"]["details"]["status_code"] == 500
