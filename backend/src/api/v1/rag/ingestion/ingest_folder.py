"""
Folder Ingestion Endpoint.

Triggers asynchronous ingestion of a local folder into the RAG system.
"""

from fastapi import APIRouter, Depends, HTTPException, Body
from pydantic import BaseModel, Field
from typing import Optional, List
import logging
import os

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.celery_worker import ingest_folder_task
from src.core.ingest_config import ChunkConfig

logger = logging.getLogger(__name__)
router = APIRouter()

class IngestFolderRequest(BaseModel):
    folder_path: str = Field(..., description="Absolute path to the folder to ingest")
    collection_name: str = Field(..., description="Name of the target collection")
    profile: str = Field("default", description="Ingestion profile (codebase, documents, default)")
    recursive: bool = Field(True, description="Whether to scan recursively")
    allowed_extensions: Optional[List[str]] = Field(None, description="Specific extensions to include")

@router.post("/ingest-folder")
async def ingest_folder_endpoint(
    request: IngestFolderRequest,
    current_user: User = Depends(get_current_user),
    rag_client=Depends(get_rag_client)
):
    """
    Start an asynchronous ingestion task for a local folder.
    
    1. Validates the folder exists.
    2. Creates a Celery task to process it.
    3. Returns the Task ID.
    """
    logger.info(f"Received folder ingestion request: {request.folder_path} -> {request.collection_name}")

    if not os.path.exists(request.folder_path):
        raise HTTPException(status_code=404, detail=f"Folder not found: {request.folder_path}")

    # Trigger Celery Task directly
    # The task itself handles scanning and processing
    task = ingest_folder_task.delay(
        folder_path=request.folder_path,
        collection_name=request.collection_name,
        profile=request.profile,
        recursive=request.recursive,
        allowed_extensions=request.allowed_extensions
    )
    
    return {
        "task_id": task.id,
        "status": "queued",
        "message": f"Started ingestion of '{request.folder_path}' into '{request.collection_name}'"
    }
