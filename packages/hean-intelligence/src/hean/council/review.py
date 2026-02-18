"""Council review data models."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Category(str, Enum):
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"
    TRADING = "trading"
    RISK = "risk"
    PERFORMANCE = "performance"


class ApprovalStatus(str, Enum):
    PENDING = "pending"
    AUTO_APPLIED = "auto_applied"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"
    FAILED = "failed"


class Recommendation(BaseModel):
    """Single recommendation from a council member."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    member_role: str
    model_id: str
    severity: Severity
    category: Category
    title: str
    description: str
    action: str
    auto_applicable: bool = False
    approval_status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    applied_at: str | None = None
    apply_result: dict[str, Any] | None = None
    target_strategy: str | None = None
    param_changes: dict[str, Any] | None = None
    code_diff: str | None = None
    target_file: str | None = None

    # Deliberation round: True если другой член совета кардинально не согласен
    # (разница severity >= 2 уровня в той же категории). Contested рекомендации
    # не применяются автоматически — требуют ручного рассмотрения.
    contested: bool = False
    contested_by: list[str] = Field(default_factory=list)  # Роли несогласных членов


class CouncilReview(BaseModel):
    """Complete review from one council member."""

    review_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    member_role: str
    model_id: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    summary: str = ""
    recommendations: list[Recommendation] = []
    raw_response: str = ""
    processing_time_ms: float = 0.0
    token_usage: dict[str, int] | None = None


class CouncilSession(BaseModel):
    """A complete council session (all members reviewed)."""

    session_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    started_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: str | None = None
    reviews: list[CouncilReview] = []
    total_recommendations: int = 0
    auto_applied_count: int = 0
    pending_approval_count: int = 0
    contested_count: int = 0  # Количество спорных рекомендаций (deliberation round)
