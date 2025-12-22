"""
nAI Core API Routes
"""

from .health import router as health_router
from .ingest import router as ingest_router
from .ask import router as ask_router
from .documents import router as documents_router
from .chat import router as chat_router
from .auth import router as auth_router

__all__ = [
    "health_router",
    "ingest_router",
    "ask_router",
    "documents_router",
    "chat_router",
    "auth_router",
]

