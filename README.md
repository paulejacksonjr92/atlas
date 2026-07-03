# Atlas

Atlas is a local, Docker-first AI platform built around Ollama, Open WebUI, Qdrant, Redis, and a small FastAPI service.

## Version

Current API version: `0.6.1`

## Project Layout

```text
C:\Atlas
+-- api
|   +-- app
|   |   +-- __init__.py
|   |   +-- main.py
|   |   +-- settings.py
|   +-- tests
|   |   +-- test_api.py
|   +-- Dockerfile
|   +-- requirements.txt
+-- compose
|   +-- docker-compose.yml
+-- data
+-- logs
+-- backups
```

The `data`, `logs`, and `backups` folders are runtime state and are intentionally ignored by Git.

## Services

| Service | Local URL |
| --- | --- |
| Atlas API | <http://localhost:8000> |
| Open WebUI | <http://localhost:3000> |
| Ollama | <http://localhost:11434> |
| Qdrant | <http://localhost:6333> |
| Redis | `localhost:6379` |

## Network Placement

Atlas runs on the isolated services VLAN with Jonas and the NAS.

Static Atlas host IP:

```text
10.87.40.69
```

Internal service URLs:

| Service | VLAN URL |
| --- | --- |
| Atlas API | <http://10.87.40.69:8000> |
| Open WebUI | <http://10.87.40.69:3000> |
| Ollama | <http://10.87.40.69:11434> |
| Qdrant | <http://10.87.40.69:6333> |
| Redis | `10.87.40.69:6379` |

Do not expose Atlas directly to the internet. Use VLAN firewall rules, VPN, or an authenticated reverse proxy for access from outside the services network.

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Service index |
| `GET /health` | API health and configuration |
| `GET /version` | API version metadata |
| `GET /status` | Service status for Atlas dependencies |
| `GET /models` | Available Ollama models |
| `POST /chat` | Non-streaming Ollama generation |
| `POST /chat/grounded` | Chat using Atlas memory and document context |
| `GET /v1/models` | OpenAI-compatible model discovery |
| `POST /v1/chat/completions` | OpenAI-compatible grounded chat for Open WebUI |
| `POST /embeddings` | Create an embedding for text |
| `POST /memory` | Store text in Atlas memory |
| `GET /memory/search` | Search Atlas memory |
| `POST /documents` | Ingest a text document into Atlas memory |
| `GET /documents/search` | Search ingested document chunks |

All successful API responses include:

```json
{
  "request_id": "uuid",
  "created_at": "utc timestamp"
}
```

Structured errors use this shape:

```json
{
  "request_id": "uuid",
  "created_at": "utc timestamp",
  "error": {
    "code": "error_code",
    "message": "Human-readable message.",
    "details": {}
  }
}
```

## Run Atlas

From the compose directory:

```powershell
cd C:\Atlas\compose
docker compose up -d --build
docker exec atlas-ollama ollama pull llama3.1:8b
docker exec atlas-ollama ollama pull nomic-embed-text
```

## Verify The Stack

```powershell
docker compose ps
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/version
curl http://localhost:8000/status
curl http://localhost:8000/models
curl http://localhost:8000/v1/models
curl -Method POST http://localhost:8000/chat -ContentType "application/json" -Body '{"prompt":"Say hello in one short sentence."}'
curl -Method POST http://localhost:8000/chat/grounded -ContentType "application/json" -Body '{"prompt":"Which VLAN are the Apple TVs assigned?","retrieval_limit":4}'
curl -Method POST http://localhost:8000/v1/chat/completions -ContentType "application/json" -Body '{"model":"atlas-grounded","messages":[{"role":"user","content":"Which VLAN are the Apple TVs assigned?"}]}'
curl -Method POST http://localhost:8000/embeddings -ContentType "application/json" -Body '{"text":"Apple TVs are assigned to VLAN 40."}'
curl -Method POST http://localhost:8000/memory -ContentType "application/json" -Body '{"text":"Apple TVs are assigned to VLAN 40.","source":"manual-note","metadata":{"client":"Los Padrinos"}}'
curl "http://localhost:8000/memory/search?query=apple%20tv%20vlan&limit=3"
curl -Method POST http://localhost:8000/documents -ContentType "application/json" -Body '{"title":"Los Padrinos Network Notes","text":"Apple TVs are assigned to VLAN 40. The MDF switch is a Cisco Catalyst.","source":"manual-document","metadata":{"client":"Los Padrinos"}}'
curl "http://localhost:8000/documents/search?query=apple%20tv%20vlan&limit=3"
```

Atlas uses `llama3.1:8b` for chat and `nomic-embed-text` for embeddings by default.

## Open WebUI

Atlas exposes an internal OpenAI-compatible API for Open WebUI.

Use this base URL from inside the Docker network:

```text
http://atlas-api:8000/v1
```

Use this base URL from the services VLAN:

```text
http://10.87.40.69:8000/v1
```

Model:

```text
atlas-grounded
```

Keep the Atlas API private. Do not expose port `8000` through Cloudflare.

## Run API Tests

From the API directory:

```powershell
cd C:\Atlas\api
python -m pip install -r requirements.txt
python -m pytest
```

## Release Notes

### Atlas API v0.6.0

- Added `GET /v1/models`.
- Added `POST /v1/chat/completions`.
- Added OpenAI-compatible response shape for Open WebUI.
- Added `atlas-grounded` model alias routed through Atlas grounded chat.
- Kept Atlas API private to the services VLAN.

### Atlas API v0.6.1

- Added OpenAI-compatible streaming responses for Open WebUI.
- `stream: true` now returns Server-Sent Events and ends with `[DONE]`.

### Atlas API v0.5.0

- Added `POST /chat/grounded`.
- Added retrieval across manual memory and document chunks.
- Added source-aware grounded prompts.
- Added grounded responses with returned source references.
- Expanded API tests for grounded chat.

### Atlas API v0.4.0

- Added `POST /documents` for text document ingestion.
- Added text chunking with chunk metadata.
- Added Qdrant-backed `GET /documents/search`.
- Added a separate `atlas_documents` collection.
- Expanded API tests for document ingestion and retrieval.

### Atlas API v0.3.0

- Added service configuration module.
- Added `GET /status` for Ollama, Qdrant, Redis, and Open WebUI checks.
- Added `POST /embeddings`.
- Added Qdrant-backed `POST /memory`.
- Added Qdrant-backed `GET /memory/search`.
- Expanded API tests for the memory foundation.

### Atlas API v0.2.0

- Added response metadata: `request_id` and `created_at`.
- Added `GET /version`.
- Added structured validation and upstream error responses.
- Added focused API tests.
- Added Git ignore rules for runtime data and local cache files.
- Added Docker Compose verification commands.

### Atlas API v0.1.2

- Working Docker stack.
- Ollama GPU inference.
- Open WebUI, Qdrant, Redis, and Atlas API running.
- Verified `/`, `/health`, `/models`, and `/chat`.
