import os

APP_VERSION = "0.7.0"
SERVICE_NAME = "atlas-api"
OPENAI_COMPAT_MODEL = "atlas-grounded"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://atlas-ollama:11434")
QDRANT_BASE_URL = os.getenv("QDRANT_BASE_URL", "http://atlas-qdrant:6333")
OPENWEBUI_BASE_URL = os.getenv("OPENWEBUI_BASE_URL", "http://atlas-openwebui:8080")
REDIS_HOST = os.getenv("REDIS_HOST", "atlas-redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.1:8b")
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
MEMORY_COLLECTION = os.getenv("MEMORY_COLLECTION", "atlas_memory")
DOCUMENT_COLLECTION = os.getenv("DOCUMENT_COLLECTION", "atlas_documents")
