# Atlas

Atlas is a local, Docker-first AI platform built around Ollama, Open WebUI, Qdrant, Redis, and a small FastAPI service.

## Version

Current API version: `0.8.11`

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
| `GET /knowledge/policy` | Show Atlas knowledge ingestion boundaries |
| `GET /knowledge/sources` | List registered document sources Atlas knows about |

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
curl -Method POST http://localhost:8000/chat/grounded -ContentType "application/json" -Body '{"prompt":"What does PatchCraft own compared to Atlas?","retrieval_limit":4}'
curl -Method POST http://localhost:8000/v1/chat/completions -ContentType "application/json" -Body '{"model":"atlas-grounded","messages":[{"role":"user","content":"What does PatchCraft own compared to Atlas?"}]}'
curl -Method POST http://localhost:8000/embeddings -ContentType "application/json" -Body '{"text":"PatchCraft is separate from Atlas."}'
curl -Method POST http://localhost:8000/memory -ContentType "application/json" -Body '{"text":"Atlas may reason over sanitized PatchCraft context.","source":"manual-note","metadata":{"project":"PatchCraft","safety":"sanitized","type":"policy"}}'
curl "http://localhost:8000/memory/search?query=PatchCraft%20Atlas&limit=3"
curl -Method POST http://localhost:8000/documents -ContentType "application/json" -Body '{"title":"PatchCraft Sanitized Overview","text":"PatchCraft is separate from Atlas. Atlas reasons over sanitized context.","source":"manual-document","metadata":{"project":"PatchCraft","safety":"sanitized","type":"overview"}}'
curl "http://localhost:8000/documents/search?query=PatchCraft%20Atlas&limit=3"
curl http://localhost:8000/knowledge/policy
curl http://localhost:8000/knowledge/sources
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

## Identity-Aware Retrieval

Atlas grounded chat accepts optional caller headers:

```text
X-Atlas-User: paul
X-Atlas-Role: admin
X-Atlas-Projects: PatchCraft,StudioServices
```

When headers are missing, Atlas uses a locked-down anonymous context. Anonymous callers can retrieve sanitized, public, or policy-level sources only. Internal roles such as `admin`, `internal`, `owner`, `operator`, `accounting`, and `tech` may retrieve reviewed internal sources, optionally scoped by `X-Atlas-Projects`.

Grounded responses include an `access` summary showing caller context, sources considered, sources allowed, and sources filtered.

## Run API Tests

From the API directory:

```powershell
cd C:\Atlas\api
python -m pip install -r requirements.txt
python -m pytest
```

## Release Notes

### Atlas API v0.8.11

- Replaced curly apostrophes in deterministic Atlas incident fallback with ASCII apostrophes to avoid Windows console encoding artifacts.
- Preserved v0.8.10 grounded endpoint fallback behavior and regression coverage.
### Atlas API v0.8.10

- Wired `atlas_direct_response` into the actual `/chat/grounded` endpoint.
- Ensured grounded responses pass through `clean_atlas_response` before returning.
- Added endpoint-level regression coverage for “what happened to Atlas?” prompts so the model cannot invent fake incident lore.
### Atlas API v0.8.9

- Added deterministic direct response for “what happened to Atlas?” style prompts to prevent fake incident reports.
- Preserved telemetry honesty by refusing to invent outage causes, log findings, API failures, sync delays, or recovery events.
- Kept distilled Atlas voice while avoiding model-generated incident lore.
### Atlas API v0.8.8

- Added cleanup for fake incident narratives, including invented reboots, outage causes, config updates, log confirmations, and system recovery claims.
- Added prompt guardrail for questions about what happened to Atlas.
- Preserved the grumpy Atlas voice while forcing unverified incidents to stay unverified.
### Atlas API v0.8.7

- Added broader cleanup for fake reboot, downtime, sync-delay, log-health, service-health, database, patching, and monitoring claims.
- Ensured known response return paths apply `clean_atlas_response`.
- Preserved Atlas personality while preventing invented operational telemetry.
### Atlas API v0.8.6

- Fixed response cleanup order so leaked mode labels are removed before whole-response quote stripping.
- Added stronger guardrails against fake monitoring, patching, metrics, log, and system-health claims.
- Preserved distilled Atlas personality while preventing fake operational status narration.
### Atlas API v0.8.5

- Added response cleanup to prevent quoted roleplay replies, leaked internal mode labels, and fake bracketed telemetry placeholders.
- Strengthened telemetry honesty language so Atlas cannot imply health checks, metrics, logs, or service status were inspected unless actually checked.
- Preserved distilled Grumpy Enlightened IT Lead behavior without cue-card leakage.
### Atlas API v0.8.4

- Distilled the Atlas persona to reduce cue-card behavior and repeated example responses.
- Added explicit rules preventing quoted replies, internal mode label leakage, example parroting, and repeated catchphrases.
- Preserved the Grumpy Enlightened IT Lead identity while reinforcing telemetry honesty and verification humility.
- Kept dual-world behavior for Paul-facing operations and client-facing workflows.
### Atlas API v0.8.3

- Added Casual Check-In Behavior to Atlas persona.
- Taught Atlas to respond warmly and dryly to casual status/vibe questions.
- Blocked fake operational claims about live health, sync freshness, monitoring visibility, or services "reporting in" unless actually verified.
- Added the core rule: vibe is allowed; fake telemetry is not.
### Atlas API v0.8.2

- Added Paul Operating Philosophy to Atlas persona.
- Added response modes for incident handling, verification, security review, memory training, architecture decisions, coding, and debugging.
- Preserved grounded retrieval, identity-aware source filtering, knowledge ingestion boundaries, and reduced source-vomit behavior.
- Tuned Atlas for dry wit, practical uncertainty handling, and Paul’s implementation-driven workflow.
### Atlas API v0.8.1

- Added Atlas persona prompt for warmer, sharper, Paul-aware grounded responses.
- Replaced rigid missing-memory phrasing with useful uncertainty handling.
- Reduced source-vomit by only appending source lists when the response actually cites sources.
- Preserved grounded retrieval, identity-aware source filtering, and knowledge ingestion boundaries.

### Atlas API v0.8.0

- Added caller context parsing from `X-Atlas-User`, `X-Atlas-Role`, and `X-Atlas-Projects`.
- Added permission-aware source filtering for grounded retrieval.
- Anonymous callers are limited to sanitized, public, and policy-level context.
- Internal roles can retrieve reviewed sources, optionally scoped by project.
- Grounded responses now include source access audit metadata.
- Removed the old incorrect network-note fixture from the API tests and README examples.

### Atlas API v0.7.0

- Added `GET /knowledge/policy` for explicit Atlas ingestion boundaries.
- Added `GET /knowledge/sources` to list registered document sources.
- Added document ingestion blocking for unsafe sources such as `.env`, logs, backups, database dumps, raw secrets, and unsafe metadata.
- Documented that Atlas stores orientation and policy context, not app-owned source-of-truth data.

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












