# Atlas

Atlas is a local, Docker-first AI platform built around Ollama, Open WebUI, Qdrant, Redis, and a small FastAPI service.

## Version

Current API version: `0.2.0`

## Project Layout

```text
C:\Atlas
├── api
│   ├── app
│   │   ├── __init__.py
│   │   └── main.py
│   ├── tests
│   │   └── test_api.py
│   ├── Dockerfile
│   └── requirements.txt
├── compose
│   └── docker-compose.yml
├── data
├── logs
└── backups
```

The `data`, `logs`, and `backups` folders are runtime state and are intentionally ignored by Git.

## Services

| Service | URL |
| --- | --- |
| Atlas API | <http://localhost:8000> |
| Open WebUI | <http://localhost:3000> |
| Ollama | <http://localhost:11434> |
| Qdrant | <http://localhost:6333> |
| Redis | `localhost:6379` |

## API Endpoints

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Service index |
| `GET /health` | API health and configuration |
| `GET /version` | API version metadata |
| `GET /models` | Available Ollama models |
| `POST /chat` | Non-streaming Ollama generation |

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
```

## Verify The Stack

```powershell
docker compose ps
curl http://localhost:8000/
curl http://localhost:8000/health
curl http://localhost:8000/version
curl http://localhost:8000/models
curl -Method POST http://localhost:8000/chat -ContentType "application/json" -Body '{"prompt":"Say hello in one short sentence."}'
```

## Run API Tests

From the API directory:

```powershell
cd C:\Atlas\api
python -m pip install -r requirements.txt
python -m pytest
```

## Release Notes

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
