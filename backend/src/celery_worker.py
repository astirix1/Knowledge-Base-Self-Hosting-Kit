"""
Celery worker for background ingestion tasks.
"""

from celery import Celery
import os
import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import SentenceSplitter

# Celery configuration
celery_app = Celery(
    "knowledge_base",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

# Ingestion profiles
PROFILES = {
    "codebase": {
        "extensions": [".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".json", ".yml", ".yaml", ".html", ".css", ".sql"],
        "chunk_size": 512,
        "chunk_overlap": 50,
        "exclude_dirs": [".git", "__pycache__", "node_modules", "venv", ".venv", "dist", "build", ".next"]
    },
    "documents": {
        "extensions": [".pdf", ".docx", ".txt", ".md"],
        "chunk_size": 800,
        "chunk_overlap": 100,
        "exclude_dirs": []
    },
    "default": {
        "extensions": [".pdf", ".docx", ".txt", ".md", ".py", ".js"],
        "chunk_size": 800,
        "chunk_overlap": 100,
        "exclude_dirs": [".git", "__pycache__", "node_modules"]
    }
}

@celery_app.task(bind=True)
def ingest_folder_task(self, folder_path, collection_name, profile="codebase", recursive=True, allowed_extensions=None):
    """
    Ingest a folder into ChromaDB.
    
    Args:
        folder_path: Path to folder
        collection_name: ChromaDB collection name
        profile: Ingestion profile (codebase, documents, default)
        recursive: Scan recursively
        allowed_extensions: Override profile extensions
    """
    # Get profile config
    config = PROFILES.get(profile, PROFILES["default"])
    extensions = allowed_extensions or config["extensions"]
    
    # Scan for files
    files = []
    for root, dirs, filenames in os.walk(folder_path):
        # Exclude directories
        dirs[:] = [d for d in dirs if d not in config["exclude_dirs"]]
        
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in extensions:
                files.append(os.path.join(root, filename))
        
        if not recursive:
            break
    
    total_files = len(files)
    if total_files == 0:
        return {"success": False, "error": "No matching files found"}
    
    # Initialize ChromaDB
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000))
    )
    
    # Get or create collection
    try:
        collection = chroma_client.get_or_create_collection(collection_name)
    except Exception as e:
        return {"success": False, "error": f"Failed to create collection: {str(e)}"}
    
    # Process files
    processed = 0
    failed = 0
    
    for i, file_path in enumerate(files):
        try:
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "progress": int((i / total_files) * 100),
                    "current_file": os.path.basename(file_path),
                    "processed": i,
                    "total": total_files
                }
            )
            
            # Read document using DoclingLoaderFactory
            from src.core.docling_loader import DoclingLoaderFactory
            loader = DoclingLoaderFactory.create_loader(file_path)
            documents = loader.load()
            
            # Chunk document
            splitter = SentenceSplitter(
                chunk_size=config["chunk_size"],
                chunk_overlap=config["chunk_overlap"]
            )
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Add to ChromaDB
            for node in nodes:
                collection.add(
                    documents=[node.text],
                    metadatas=[{"source": file_path, "chunk_id": node.id_}],
                    ids=[node.id_]
                )
            
            processed += 1
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            failed += 1
    
    return {
        "success": True,
        "processed_files": processed,
        "failed_files": failed,
        "total_files": total_files,
        "collection": collection_name
    }


@celery_app.task(bind=True)
def ingest_batch_task(self, assignments, chunk_size=512, chunk_overlap=50):
    """
    Ingest a batch of files into specific collections.
    
    Args:
        assignments: List of dicts with 'file_path' and 'collection'
        chunk_size: Chunk size
        chunk_overlap: Chunk overlap
    """
    # Initialize ChromaDB
    chroma_client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000))
    )
    
    total_files = len(assignments)
    processed = 0
    failed = 0
    
    # Cache collections to avoid repeated get_or_create
    collections = {}
    
    for i, assignment in enumerate(assignments):
        file_path = assignment["file_path"]
        collection_name = assignment["collection"]
        
        try:
            # Update progress
            self.update_state(
                state="PROCESSING",
                meta={
                    "progress": int((i / total_files) * 100),
                    "current_file": os.path.basename(file_path),
                    "processed": i,
                    "total": total_files
                }
            )
            
            # Get collection
            if collection_name not in collections:
                collections[collection_name] = chroma_client.get_or_create_collection(collection_name)
            collection = collections[collection_name]
            
            # Read document using DoclingLoaderFactory
            from src.core.docling_loader import DoclingLoaderFactory
            loader = DoclingLoaderFactory.create_loader(file_path)
            documents = loader.load()
            
            # Chunk document
            splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            nodes = splitter.get_nodes_from_documents(documents)
            
            # Add to ChromaDB
            for node in nodes:
                collection.add(
                    documents=[node.text],
                    metadatas=[{"source": file_path, "chunk_id": node.id_}],
                    ids=[node.id_]
                )
            
            processed += 1
            
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")
            failed += 1
            
    return {
        "success": True,
        "processed_files": processed,
        "failed_files": failed,
        "total_files": total_files
    }
