"""
Feedback Service for storing user feedback and calculating metrics.
"""

from typing import Dict, Any


class FeedbackService:
    def __init__(self):
        self.positive_feedback = 0
        self.negative_feedback = 0
        self.total_feedback = 0

    async def add_feedback(self, is_positive: bool) -> Dict[str, Any]:
        """Record user feedback."""
        if is_positive:
            self.positive_feedback += 1
        else:
            self.negative_feedback += 1

        self.total_feedback += 1

        return {"status": "success", "current_accuracy": self.calculate_accuracy()}

    def calculate_accuracy(self) -> float:
        """
        Calculate accuracy based on feedback.
        Formula: (Positive Feedback / Total Feedback) * 100
        Default: 100.0 if no feedback.
        """
        if self.total_feedback == 0:
            return 100.0

        return (self.positive_feedback / self.total_feedback) * 100.0

    def get_stats(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return {
            "positive": self.positive_feedback,
            "negative": self.negative_feedback,
            "total": self.total_feedback,
            "accuracy": self.calculate_accuracy(),
        }
