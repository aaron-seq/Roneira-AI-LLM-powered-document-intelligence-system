"""Main application package for Document Intelligence System.

Enterprise-grade document processing platform with AI-powered insights
and modern cloud-native architecture.
"""

__version__ = "2.0.0"
__author__ = "Aaron Sequeira"
__description__ = "Document Intelligence System with Azure AI integration"

# Import main application for easy access
from .main import app, create_application

__all__ = [
    "app",
    "create_application",
    "__version__",
    "__author__",
    "__description__"
]
