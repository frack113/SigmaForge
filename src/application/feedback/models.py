"""Feedback models and schemas."""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field

from src.shared.utils.crypto_utils import compute_sha256_str


class FeedbackIn(BaseModel):
    """Input schema for feedback submission."""

    query: str = Field(..., description="The search query that feedback relates to")
    helpful: bool = Field(..., description="Whether the results were helpful")


class Feedback(BaseModel):
    """Feedback database model."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique feedback ID")
    query_hash: str = Field(..., description="Hashed query for anonymity")
    helpful: bool = Field(..., description="Whether results were helpful")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Feedback timestamp")
    session_id: str | None = Field(default=None, description="Session ID for tracking")


class FeedbackResponse(BaseModel):
    """Response schema for feedback submission."""

    feedback_id: str
    message: str = "Feedback recorded successfully"


class FeedbackStats(BaseModel):
    """Feedback statistics for admin."""

    total: int
    helpful_count: int
    not_helpful_count: int
    helpful_percentage: float


def hash_query(query: str) -> str:
    """Generate a hash of the query for anonymity."""
    return compute_sha256_str(query)[:16]
