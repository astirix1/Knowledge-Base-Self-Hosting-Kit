"""
Feature limits for different editions - Community Edition Reference

This is a simplified reference showing the edition-based architecture.
The full production version includes additional enterprise features and
dynamic license validation.

For production deployment with full feature set, contact sales.
"""

from typing import Dict, Any, Optional
from enum import Enum
import os


class Edition(Enum):
    """Application edition tiers."""
    COMMUNITY = "community"      # Open source, basic features
    PROFESSIONAL = "professional"  # Extended features (contact sales)
    ENTERPRISE = "enterprise"     # Full features + support (contact sales)


class FeatureLimits:
    """
    Feature limits based on edition tier.

    Community Edition: Basic RAG functionality for evaluation
    Professional/Enterprise: Contact sales for full feature set

    NOTE: This is a reference implementation showing the architecture.
    Production versions include:
    - Dynamic license validation
    - Usage-based metering
    - Custom feature bundles
    - SSO and RBAC integration
    """

    # Community Edition Limits (evaluation/demo purposes)
    COMMUNITY_LIMITS = {
        # Collection limits
        "max_collections": 3,
        "max_documents_per_collection": 1000,
        "max_total_documents": 3000,

        # File format limits
        "allowed_file_formats": [".pdf", ".txt", ".md"],
        "max_file_size_mb": 10,

        # Feature flags
        "enable_advanced_rag": False,  # Basic retrieval only
        "enable_hybrid_search": True,   # Vector + keyword search included
        "enable_reranking": False,      # Advanced reranking: Professional+
        "enable_multi_collection": False,  # Multi-collection query: Professional+
        "enable_batch_ingestion": True,    # Batch processing included
        "enable_custom_embeddings": False, # Custom models: Enterprise only

        # Performance limits
        "max_concurrent_queries": 2,
        "query_timeout_seconds": 30,

        # API rate limits (per hour)
        "api_query_limit": 1000,

        # Edition metadata
        "edition_name": "Community",
        "support_level": "community",
        "upgrade_url": "https://your-domain.com/pricing"
    }

    # Note: Professional and Enterprise limits are defined in the
    # production version with license key validation.
    # Contact sales for details: [your-contact-email]

    @classmethod
    def get_limits(cls, edition: Edition = None) -> Dict[str, Any]:
        """
        Get feature limits for specified edition.

        Community Edition: Returns public limits
        Professional/Enterprise: Requires license validation (production only)

        Args:
            edition: Edition tier (defaults to Community)

        Returns:
            Dict of feature limits
        """
        if edition is None or edition == Edition.COMMUNITY:
            return cls.COMMUNITY_LIMITS.copy()

        # Professional/Enterprise require license validation
        # This is a stub - production version validates licenses
        return {
            **cls.COMMUNITY_LIMITS,
            "license_required": True,
            "contact": "Contact sales for Professional/Enterprise features"
        }

    @classmethod
    def check_limit(cls, feature: str, current_value: int, edition: Edition = None) -> bool:
        """
        Check if current value is within limits for edition.

        Args:
            feature: Feature name (e.g., 'max_collections')
            current_value: Current usage value
            edition: Edition tier

        Returns:
            True if within limits, False otherwise
        """
        limits = cls.get_limits(edition)
        max_value = limits.get(feature, 0)

        # -1 means unlimited (Enterprise feature)
        if max_value == -1:
            return True

        return current_value < max_value

    @classmethod
    def is_feature_enabled(cls, feature: str, edition: Edition = None) -> bool:
        """
        Check if a feature is enabled for edition.

        Args:
            feature: Feature flag name (e.g., 'enable_advanced_rag')
            edition: Edition tier

        Returns:
            True if feature is enabled
        """
        limits = cls.get_limits(edition)
        return limits.get(feature, False)

    @classmethod
    def get_edition_from_env(cls) -> Edition:
        """
        Get edition from environment variable.

        Community Edition is default for self-hosted deployments.
        Production versions support license key validation.

        Returns:
            Edition enum value
        """
        edition_str = os.getenv("EDITION", "community").lower()

        try:
            return Edition(edition_str)
        except ValueError:
            # Default to community for invalid values
            return Edition.COMMUNITY


# Convenience function for common use case
def get_current_edition() -> Edition:
    """Get the currently configured edition."""
    return FeatureLimits.get_edition_from_env()
