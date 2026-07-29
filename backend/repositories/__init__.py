"""Data access layer.

Repositories are the only place that talks to the ORM. Services depend on
them, never on sessions directly, so the storage strategy can change without
touching business logic.
"""

from backend.repositories.document_repository import DocumentRepository
from backend.repositories.feedback_repository import FeedbackRepository

__all__ = ["DocumentRepository", "FeedbackRepository"]
