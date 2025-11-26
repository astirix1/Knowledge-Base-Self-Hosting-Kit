"""
Base Generator for LLM Tasks.

This is a simplified reference implementation showing the generator architecture.
The full production version with advanced prompt engineering, security hardening,
and optimized LLM interactions is available in the Enterprise Edition.

For inquiries about the full implementation, contact: [your-contact]
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """
    Abstract base class for LLM-powered generators.

    This is a reference implementation demonstrating the architecture.
    Production features include:
    - Advanced prompt injection protection
    - Multi-turn conversation handling
    - Token optimization strategies
    - Fallback mechanisms
    - Cost tracking and rate limiting

    NOTE: This simplified version provides basic functionality.
    See Enterprise Edition for production-grade features.
    """

    def __init__(self, llm_singleton, data_classifier_service=None, settings_service=None):
        """
        Initialize generator with required services.

        Args:
            llm_singleton: LLM client provider
            data_classifier_service: Optional data classification service
            settings_service: Optional settings service
        """
        self.llm_client = llm_singleton.get_client() if llm_singleton else None
        self.classifier = data_classifier_service
        self.settings_service = settings_service

        logger.info(f"{self.__class__.__name__} initialized (Community Edition)")

    @abstractmethod
    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output based on input data.

        Subclasses must implement this method.

        Args:
            input_data: Input parameters for generation

        Returns:
            Dict containing generated content and metadata
        """
        pass

    def _validate_folder_path(self, folder_path: str) -> bool:
        """Basic folder path validation."""
        import os
        return os.path.exists(folder_path) and os.path.isdir(folder_path)

    def _sanitize_prompt(self, prompt: str) -> str:
        """
        Basic prompt sanitization.

        NOTE: Production version includes advanced security measures:
        - Multi-layer prompt injection detection
        - Context-aware sanitization
        - Token budget management
        - Jailbreak attempt detection
        """
        # Basic sanitization only
        return prompt.strip()

    async def _call_llm(self, prompt: str, **kwargs) -> str:
        """
        Simplified LLM call.

        NOTE: Production version includes:
        - Retry logic with exponential backoff
        - Circuit breaker pattern
        - Response validation
        - Cost optimization
        - Streaming support
        """
        if not self.llm_client:
            raise ValueError("LLM client not configured")

        # Basic call - production version has much more sophistication
        try:
            response = await self.llm_client.generate(prompt, **kwargs)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
