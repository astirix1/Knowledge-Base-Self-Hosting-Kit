"""
Database Models - Community Edition Stub

Community Edition uses in-memory storage only.
Enterprise Edition includes PostgreSQL/SQLite with full persistence.
"""

from typing import Optional, Dict, Any
from datetime import datetime

class DummyUser:
    """Dummy user model for Community Edition"""
    id: int = 1
    username: str = "community_user"
    email: str = "user@community.local"
    is_active: bool = True

# Alias for compatibility
User = DummyUser

class CollectionIndexConfig:
    """Dummy collection config for Community Edition"""
    id: Optional[int] = None
    collection_name: str = ""
    user_id: int = 1
    config_data: Dict[str, Any] = {}
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()
    
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
