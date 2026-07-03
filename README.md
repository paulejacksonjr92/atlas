# Atlas

Atlas is a local, Docker-first AI platform built around Ollama, Open WebUI, Qdrant, Redis, and a small FastAPI service.

## Version

Current API version: `0.3.0`

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
| `POST /embeddings` | Create an embedding for text |
| `POST /memory` | Store text in Atlas memory |
| `GET /memory/search` | Search Atlas memory |

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
curl -Method POST http://localhost:8000/chat -ContentType "application/json" -Body '{"prompt":"Say hello in one short sentence."}'
curl -Method POST http://localhost:8000/embeddings -ContentType "application/json" -Body '{"text":"Apple TVs are assigned to VLAN 40."}'
curl -Method POST http://localhost:8000/memory -ContentType "application/json" -Body '{"text":"Apple TVs are assigned to VLAN 40.","source":"manual-note","metadata":{"client":"Los Padrinos"}}'
curl "http://localhost:8000/memory/search?query=apple%20tv%20vlan&limit=3"
```

Atlas uses `llama3.1:8b` for chat and `nomic-embed-text` for embeddings by default.

## Run API Tests

From the API directory:

```powershell
cd C:\Atlas\api
python -m pip install -r requirements.txt
python -m pytest
```

## Release Notes

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
