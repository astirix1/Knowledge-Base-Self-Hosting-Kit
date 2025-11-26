"""
Ragignore Generator - Community Edition

Generates .ragignore files for intelligent folder filtering.

This is a simplified reference implementation. The full production version includes:
- Advanced pattern learning from codebase analysis
- Multi-language project detection
- Framework-specific exclusion patterns
- Intelligent size-based filtering
- Historical pattern optimization

For the full implementation, see Enterprise Edition.
"""

from typing import Dict, Any
import logging
from src.services.generators.base_generator import BaseGenerator

logger = logging.getLogger(__name__)


class RagignoreGenerator(BaseGenerator):
    """
    Generator that creates .ragignore files based on folder analysis.

    Community Edition: Basic pattern generation
    Enterprise Edition: AI-powered pattern optimization with learning
    """

    async def generate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a .ragignore file.

        This is a basic implementation with common patterns.
        Production version uses advanced LLM-powered analysis.

        Args:
            input_data: Must contain 'folder_path' and optional 'analysis'

        Returns:
            Dict with 'status', 'content', and 'message'
        """
        folder_path = input_data.get("folder_path")

        if not folder_path or not self._validate_folder_path(folder_path):
            return {
                "status": "error",
                "message": "Invalid folder path"
            }

        # Basic default patterns (production version generates optimized patterns)
        default_patterns = [
            "# Generated .ragignore file",
            "# Note: This is a basic pattern set. Enterprise Edition provides",
            "# intelligent, project-specific pattern generation.",
            "",
            "# Dependencies",
            "node_modules/",
            "__pycache__/",
            "venv/",
            ".venv/",
            "",
            "# Build outputs",
            "dist/",
            "build/",
            "*.egg-info/",
            "",
            "# IDE",
            ".vscode/",
            ".idea/",
            "",
            "# Logs and temp",
            "*.log",
            "tmp/",
            "",
            "# Large files",
            "*.mp4",
            "*.avi",
            "*.zip",
            "*.tar.gz"
        ]

        logger.info("Generated basic .ragignore (Community Edition)")
        logger.info("Enterprise Edition: AI-powered pattern optimization available")

        return {
            "status": "success",
            "content": "\n".join(default_patterns),
            "message": "Basic .ragignore generated. Upgrade to Enterprise for intelligent optimization.",
            "edition": "community"
        }
