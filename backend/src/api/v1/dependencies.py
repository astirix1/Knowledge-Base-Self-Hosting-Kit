"""
FastAPI dependencies for RAG service injection.
Community Edition - RAG only, no email/database features.
"""

from fastapi import Depends
from loguru import logger

from src.core.rag_client import RAGClient
from src.core.services.query_service import QueryService
from src.services.config_service import config_service

# Singleton for RAGClient (prevents connection leak)
_rag_client = None

async def get_rag_client() -> RAGClient:
    """Get RAG Client (singleton to prevent connection leak)"""
    global _rag_client
    if _rag_client is None:
        config = config_service.load_configuration()
        _rag_client = RAGClient(config=config)
        logger.info("RAGClient singleton initialized")
    return _rag_client

async def get_query_service(
    rag_client: RAGClient = Depends(get_rag_client)
) -> QueryService:
    """Get QueryService from RAGClient"""
    return rag_client.query_service
