"""
Document Upload Endpoints.

Handles file uploads and ingestion into ChromaDB collections using the Central Docling Service.
Features:
- Multi-file upload with validation
- Centralized Docling processing (Repair -> Convert -> Refine)
- Direct Markdown chunking
"""

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    File,
    UploadFile,
    Form,
    BackgroundTasks,
    status
)
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import os
import shutil
from pathlib import Path
import asyncio

from src.api.v1.dependencies import get_rag_client
from src.services.auth_service import get_current_user
from src.database.models import User
from src.services.ingestion_task_manager import ingestion_task_manager
from src.services.docling_service import docling_service
from src.core.exceptions import (
    CollectionNotFoundError,
    ValidationError,
    InvalidFileTypeError,
    ChromaDBError,
    IngestionError,
    ServiceUnavailableError
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested metadata for ChromaDB compatibility.

    ChromaDB only accepts str, int, float, bool, or None as metadata values.
    This function converts nested dicts and lists to JSON strings.
    """
    flattened = {}
    for key, value in metadata.items():
        if value is None:
            flattened[key] = None
        elif isinstance(value, (str, int, float, bool)):
            flattened[key] = value
        elif isinstance(value, (dict, list)):
            # Convert nested structures to JSON string
            import json
            flattened[key] = json.dumps(value)
        else:
            # Convert other types to string
            flattened[key] = str(value)
    return flattened


@router.post("/validate-upload")
async def validate_upload(
    collection_name: str = Form(...),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Validate if upload is compatible with collection embedding configuration."""
    logger.debug(f"Validating upload for collection: {collection_name}")

    try:
        metadata = await rag_client.get_collection_metadata(collection_name)

        if not metadata:
            logger.warning(f"No metadata for '{collection_name}', allowing upload with warning")
            return {
                "valid": True,
                "warning": "No metadata found for this collection. Upload may fail if embedding models don't match.",
                "collection_model": "unknown",
                "current_model": "unknown"
            }

        from src.services.config_service import config_service
        config = config_service.load_configuration()

        current_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
        collection_model = metadata.get("embedding_model")
        collection_dims = metadata.get("embedding_dimensions")

        try:
            current_dims = await rag_client.get_embedding_dimensions(current_model)
        except Exception:
            current_dims = 768

        if current_model != collection_model or current_dims != collection_dims:
            logger.warning(f"Embedding mismatch: {current_model} vs {collection_model}")
            raise ValidationError(
                "Embedding mismatch - collection and current model incompatible",
                details={
                    "collection_model": collection_model,
                    "current_model": current_model,
                    "collection_dims": collection_dims,
                    "current_dims": current_dims,
                    "suggestion": "Please change EMBEDDING_MODEL in Settings to match the collection"
                }
            )

        logger.debug("Upload validation passed")
        return {
            "valid": True,
            "message": "Upload is compatible",
            "collection_model": collection_model,
            "current_model": current_model
        }

    except ValidationError:
        raise  # Re-raise validation errors
    except Exception as e:
        logger.error(f"Upload validation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("validation", str(e))


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    collection_name: str = Form("default"),
    chunk_size: int = Form(500),
    chunk_overlap: int = Form(50),
    rag_client=Depends(get_rag_client),
    current_user: User = Depends(get_current_user)
):
    """Upload and index documents to ChromaDB with embedding validation."""
    logger.debug(f"Uploading {len(files)} files to collection '{collection_name}'")

    try:
        # STEP 1: Validate embedding compatibility
        metadata = await rag_client.get_collection_metadata(collection_name)

        if metadata:
            from src.services.config_service import config_service
            config = config_service.load_configuration()
            current_model = config.get("EMBEDDING_MODEL", "nomic-embed-text:latest")
            collection_model = metadata.get("embedding_model")
            collection_dims = metadata.get("embedding_dimensions")

            try:
                current_dims = await rag_client.get_embedding_dimensions(current_model)
            except Exception:
                current_dims = 768

            if current_model != collection_model or current_dims != collection_dims:
                logger.error(f"Embedding mismatch: {current_model} vs {collection_model}")
                raise ValidationError(
                    f"Embedding mismatch! Collection requires '{collection_model}' ({collection_dims} dims), but current settings use '{current_model}' ({current_dims} dims).",
                    details={
                        "collection_model": collection_model,
                        "current_model": current_model,
                        "collection_dims": collection_dims,
                        "current_dims": current_dims
                    }
                )

        results = []
        total_chunks = 0

        temp_dir = "/tmp/rag_upload"
        os.makedirs(temp_dir, exist_ok=True)

        try:
            for file in files:
                try:
                    file_path = os.path.join(temp_dir, file.filename)

                    with open(file_path, "wb") as buffer:
                        content = await file.read()
                        buffer.write(content)

                    # Use Central Docling Service
                    # This handles Repair, Analysis, and Conversion (OCR/Tables)
                    process_result = await docling_service.process_file(file_path)
                    
                    if not process_result["success"]:
                        raise IngestionError(f"Docling processing failed: {process_result.get('error')}")

                    markdown_content = process_result["content"]
                    doc_metadata = process_result["metadata"]
                    
                    # Add ingestion metadata
                    doc_metadata.update({
                        'source': file.filename,
                        'file_type': Path(file.filename).suffix.lower(),
                        'collection': collection_name,
                        'processed_by': 'docling_service_v1'
                    })

                    # Use IndexingService via RAGClient
                    # This ensures both Vector Store and BM25 Index are updated
                    from src.core.indexing_service import Document, ChunkConfig

                    # Create Document object
                    doc = Document(
                        content=markdown_content,
                        metadata=doc_metadata
                    )

                    # Create ChunkConfig
                    chunk_config = ChunkConfig(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )

                    # Index using RAGClient (which uses IndexingService)
                    response = await rag_client.index_documents(
                        documents=[doc],
                        collection_name=collection_name,
                        chunk_config=chunk_config
                    )

                    if not response.is_success:
                         raise IngestionError(f"Indexing failed: {response.error}")
                    
                    # Extract stats from response
                    chunk_count = response.data.get("indexed_nodes", 0)
                    total_chunks += chunk_count
                    
                    results.append({
                        "filename": file.filename,
                        "success": True,
                        "chunks": chunk_count
                    })

                    logger.info(f"Successfully uploaded and indexed '{file.filename}': {chunk_count} chunks")



                except Exception as file_error:
                    logger.error(f"Failed to upload '{file.filename}': {file_error}")
                    results.append({
                        "filename": file.filename,
                        "success": False,
                        "chunks": 0,
                        "error": str(file_error)
                    })

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        successful_files = sum(1 for r in results if r["success"])
        logger.info(f"Upload complete: {successful_files}/{len(files)} files, {total_chunks} chunks")

        return {
            "success": successful_files > 0,
            "files_processed": successful_files,
            "total_files": len(files),
            "total_chunks": total_chunks,
            "results": results,
            "collection": collection_name,
            "chunk_config": {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        }

    except (ValidationError, ChromaDBError, ServiceUnavailableError):
        raise  # Re-raise custom exceptions
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        raise IngestionError(str(e))
