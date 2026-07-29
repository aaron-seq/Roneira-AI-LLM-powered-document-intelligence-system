"""Persistence for answer quality feedback."""

from __future__ import annotations

import uuid
from typing import Any, Dict, Optional

from sqlalchemy import func, select

from backend.core.database import DatabaseManager
from backend.core.models import ChatFeedback


class FeedbackRepository:
    """Stores thumbs up/down votes on assistant answers.

    Keeping votes as rows (rather than a pair of in-memory counters) means the
    quality signal survives restarts and can be sliced by session, user, or
    time window when evaluating retrieval changes.
    """

    def __init__(self, db_manager: DatabaseManager):
        self._db = db_manager

    async def add(
        self,
        *,
        message_id: str,
        is_positive: bool,
        session_id: Optional[str] = None,
        owner_id: Optional[str] = None,
        comment: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Record one vote and return the updated aggregate."""
        async with self._db.get_session() as session:
            session.add(
                ChatFeedback(
                    id=str(uuid.uuid4()),
                    message_id=message_id,
                    session_id=session_id,
                    owner_id=owner_id,
                    is_positive=is_positive,
                    comment=comment[:2000] if comment else None,
                )
            )
        return await self.stats(owner_id=owner_id)

    async def stats(self, owner_id: Optional[str] = None) -> Dict[str, Any]:
        """Positive/negative counts and the satisfaction rate.

        Returns ``satisfaction_rate: None`` when no votes exist, so the UI can
        say "not rated yet" instead of displaying a fabricated score.
        """
        async with self._db.get_session() as session:
            stmt = select(ChatFeedback.is_positive, func.count()).group_by(
                ChatFeedback.is_positive
            )
            if owner_id is not None:
                stmt = stmt.where(ChatFeedback.owner_id == owner_id)

            counts = {bool(row[0]): row[1] for row in (await session.execute(stmt)).all()}

        positive = counts.get(True, 0)
        negative = counts.get(False, 0)
        total = positive + negative

        return {
            "positive": positive,
            "negative": negative,
            "total": total,
            "satisfaction_rate": round(positive / total, 4) if total else None,
        }
