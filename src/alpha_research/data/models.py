"""
Data models for the Alpha Research Trading System.

All models use Pydantic for validation and serialization.
These are the canonical data structures used throughout the system.
"""

from datetime import datetime, date as date_type
from typing import Any, Dict, List, Optional, Set, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import json

from alpha_research.utils.enums import (
    ActionType,
    EvidenceType,
    NewsFlag,
    FilingFlag,
    InsiderFlag,
    ProposalRejectReason,
)


class BaseRecord(BaseModel):
    """Base class for all data records with common fields."""

    model_config = ConfigDict(extra="allow")


# =============================================================================
# Market Data Models
# =============================================================================

class MarketData(BaseRecord):
    """Daily market data for a single security."""

    symbol: str = Field(..., description="Stock ticker symbol")
    trade_date: date_type = Field(..., description="Trading date")
    open: float = Field(..., ge=0, description="Opening price")
    high: float = Field(..., ge=0, description="High price")
    low: float = Field(..., ge=0, description="Low price")
    close: float = Field(..., ge=0, description="Closing price")
    volume: int = Field(..., ge=0, description="Trading volume")
    adj_close: Optional[float] = Field(None, ge=0, description="Adjusted close")
    vwap: Optional[float] = Field(None, ge=0, description="Volume-weighted average price")

    # Derived fields
    adv_dollar_20d: Optional[float] = Field(None, description="20-day average dollar volume")
    adv_dollar_60d: Optional[float] = Field(None, description="60-day average dollar volume")
    volatility_20d: Optional[float] = Field(None, description="20-day realized volatility")
    volatility_60d: Optional[float] = Field(None, description="60-day realized volatility")

    # Metadata
    asof_time: datetime = Field(..., description="Snapshot timestamp")
    available_at: datetime = Field(..., description="When data became available")
    content_hash: Optional[str] = Field(None, description="Hash of this record")


class FundamentalData(BaseRecord):
    """Fundamental data for a single security (point-in-time)."""

    symbol: str = Field(..., description="Stock ticker symbol")
    fiscal_period: str = Field(..., description="Fiscal period (e.g., '2024Q3')")
    report_date: date_type = Field(..., description="Report date")

    # Balance sheet
    total_assets: Optional[float] = Field(None)
    total_liabilities: Optional[float] = Field(None)
    total_debt: Optional[float] = Field(None)
    total_equity: Optional[float] = Field(None)
    book_value: Optional[float] = Field(None)
    cash_and_equivalents: Optional[float] = Field(None)

    # Income statement
    revenue: Optional[float] = Field(None)
    gross_profit: Optional[float] = Field(None)
    operating_income: Optional[float] = Field(None)
    net_income: Optional[float] = Field(None)
    ebitda: Optional[float] = Field(None)

    # Cash flow
    cfo: Optional[float] = Field(None, description="Cash from operations")
    capex: Optional[float] = Field(None)
    fcf: Optional[float] = Field(None, description="Free cash flow")

    # Per share
    eps: Optional[float] = Field(None)
    book_value_per_share: Optional[float] = Field(None)
    shares_outstanding: Optional[float] = Field(None)

    # Ratios (computed or provided)
    return_on_equity: Optional[float] = Field(None)
    return_on_assets: Optional[float] = Field(None)
    gross_profit_margin: Optional[float] = Field(None)
    operating_profit_margin: Optional[float] = Field(None)
    debt_to_assets: Optional[float] = Field(None)
    debt_to_equity: Optional[float] = Field(None)
    cfo_to_assets: Optional[float] = Field(None)
    fcf_to_assets: Optional[float] = Field(None)

    # Valuation
    market_cap: Optional[float] = Field(None)
    enterprise_value: Optional[float] = Field(None)
    ebitda_to_ev: Optional[float] = Field(None)
    book_to_price: Optional[float] = Field(None)
    earnings_to_price: Optional[float] = Field(None)

    # Sector/Industry
    sector: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)

    # Point-in-time fields (CRITICAL for no look-ahead)
    asof_time: datetime = Field(..., description="Snapshot timestamp")
    available_at: datetime = Field(..., description="When this data became available")
    content_hash: Optional[str] = Field(None, description="Hash of this record")

    @model_validator(mode='after')
    def validate_available_at_timing(self):
        """Ensure available_at is not after asof_time."""
        if self.available_at and self.asof_time and self.available_at > self.asof_time:
            raise ValueError(f"available_at ({self.available_at}) cannot be after asof_time ({self.asof_time})")
        return self


class UniverseRecord(BaseRecord):
    """A single security in the tradeable universe."""

    symbol: str = Field(..., description="Stock ticker symbol")
    exchange: str = Field(..., description="Exchange (NYSE, NASDAQ, AMEX)")
    security_type: str = Field(..., description="Security type")
    name: Optional[str] = Field(None, description="Company name")
    sector: Optional[str] = Field(None)
    industry: Optional[str] = Field(None)

    # Filters passed
    close: float = Field(..., ge=0)
    market_cap: Optional[float] = Field(None, ge=0)
    adv_dollar_60d: float = Field(..., ge=0)

    # Metadata
    universe_id: str = Field(..., description="Universe snapshot ID")
    asof_time: datetime = Field(...)
    available_at: datetime = Field(...)
    content_hash: Optional[str] = Field(None)


# =============================================================================
# Evidence Models
# =============================================================================

class Evidence(BaseRecord):
    """
    A piece of evidence that can be cited by proposals.

    All proposals must reference evidence_ids that exist in the ledger.
    Evidence must have available_at <= asof_time to be valid.
    """

    evidence_id: str = Field(..., description="Unique evidence identifier")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    symbol: str = Field(..., description="Related stock symbol")

    # Content
    content: str = Field(..., description="Evidence content text")
    source: str = Field(..., description="Source of evidence")
    source_url: Optional[str] = Field(None, description="URL to original source")

    # Timestamps (CRITICAL for point-in-time)
    published_at: datetime = Field(..., description="When content was published")
    available_at: datetime = Field(..., description="When we received it")
    asof_time: datetime = Field(..., description="Snapshot timestamp")

    # Hash for verification
    content_hash: str = Field(..., description="Hash of content")

    # Optional metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode='after')
    def validate_timestamps(self):
        """Ensure point-in-time integrity for evidence timestamps."""
        if self.available_at and self.asof_time and self.available_at > self.asof_time:
            raise ValueError(
                f"Evidence available_at ({self.available_at}) cannot be after asof_time ({self.asof_time}). "
                "This would be a point-in-time violation."
            )
        if self.published_at and self.available_at and self.published_at > self.available_at:
            raise ValueError(
                f"Evidence published_at ({self.published_at}) cannot be after available_at ({self.available_at})"
            )
        return self


class NewsEvidence(Evidence):
    """Evidence from news articles."""

    evidence_type: EvidenceType = EvidenceType.NEWS_ARTICLE
    headline: Optional[str] = Field(None)
    summary: Optional[str] = Field(None)


class FilingEvidence(Evidence):
    """Evidence from SEC filings."""

    evidence_type: EvidenceType = EvidenceType.SEC_FILING_CHUNK
    filing_type: str = Field(..., description="10-K, 10-Q, 8-K, etc.")
    section: Optional[str] = Field(None, description="Section of the filing")
    chunk_index: Optional[int] = Field(None, description="Chunk number in filing")


class InsiderEvidence(Evidence):
    """Evidence from insider trading (Form 4)."""

    evidence_type: EvidenceType = EvidenceType.INSIDER_FORM4_RECORD
    insider_name: str = Field(...)
    insider_title: Optional[str] = Field(None)
    transaction_type: str = Field(...)  # P=Purchase, S=Sale
    shares: int = Field(...)
    price: float = Field(...)
    value: float = Field(...)


# =============================================================================
# Proposal Models
# =============================================================================

class Proposal(BaseRecord):
    """
    A proposal from an LLM or rule-based module.

    Proposals must:
    1. Reference valid evidence_ids
    2. Use allowed action_types
    3. Stay within score limits
    4. Have confidence >= threshold
    """

    proposal_id: str = Field(..., description="Unique proposal identifier")
    module_name: str = Field(..., description="Name of proposing module")
    symbol: str = Field(..., description="Stock symbol")
    run_id: str = Field(..., description="Current run identifier")

    # Action
    action_type: ActionType = Field(..., description="Proposed action")
    score: float = Field(0.0, description="Score adjustment (positive or negative)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in proposal")

    # Flags
    flags: List[str] = Field(default_factory=list, description="Detected flags")

    # Evidence (REQUIRED - uncited proposals are invalid)
    evidence_ids_used: List[str] = Field(..., min_length=1, description="Evidence IDs cited")

    # Optional overrides
    position_cap: Optional[float] = Field(None, ge=0, le=1, description="Max position cap")
    delay_trade_cycles: Optional[int] = Field(None, ge=0, description="Cycles to delay")
    uncertainty_score: Optional[float] = Field(None, ge=0, le=1)

    # Validity
    valid_until: datetime = Field(..., description="Proposal expiry time")

    # Metadata
    reasoning: Optional[str] = Field(None, description="Brief reasoning")
    raw_output: Optional[str] = Field(None, description="Raw LLM output")

    # Validation status (filled by validator)
    is_valid: bool = Field(True)
    reject_reasons: List[ProposalRejectReason] = Field(default_factory=list)

    @field_validator('evidence_ids_used')
    @classmethod
    def validate_has_evidence(cls, v):
        """Ensure at least one evidence is cited."""
        if not v:
            raise ValueError("Proposal must cite at least one evidence_id")
        return v


class ValidatedProposal(Proposal):
    """A proposal that has passed validation."""

    original_score: float = Field(..., description="Score before budget adjustment")
    adjusted_score: float = Field(..., description="Score after budget cap")
    budget_applied: bool = Field(False, description="Whether budget cap was applied")


# =============================================================================
# Snapshot Models
# =============================================================================

class Snapshot(BaseRecord):
    """
    A complete snapshot of all inputs for a run.

    This is the atomic unit for deterministic replay.
    Same snapshot_id must always produce same target_weights.
    """

    snapshot_id: str = Field(..., description="Unique snapshot identifier")
    run_type: str = Field(..., description="daily, weekly, adhoc")
    asof_time: datetime = Field(..., description="Snapshot timestamp")

    # Component hashes
    universe_hash: str = Field(..., description="Hash of universe")
    market_hash: str = Field(..., description="Hash of market data")
    fundamental_hash: str = Field(..., description="Hash of fundamental data")
    evidence_hash: str = Field(..., description="Hash of evidence pack")

    # Counts
    universe_size: int = Field(...)
    evidence_count: int = Field(...)

    # Source versions
    source_versions: Dict[str, str] = Field(default_factory=dict)

    # Overall hash
    content_hash: str = Field(..., description="Hash of entire snapshot")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RunResult(BaseRecord):
    """Result of a complete run for auditing."""

    run_id: str = Field(...)
    snapshot_id: str = Field(...)
    asof_time: datetime = Field(...)

    # Inputs used
    universe_count: int = Field(...)
    candidates_count: int = Field(...)
    evidence_count: int = Field(...)

    # Proposals
    proposals_received: int = Field(...)
    proposals_valid: int = Field(...)
    proposals_rejected: int = Field(...)

    # Outputs
    holdings_count: int = Field(...)
    turnover_pct: float = Field(...)

    # Risk gate
    risk_action: str = Field(...)
    risk_scale: float = Field(...)

    # Execution
    orders_generated: int = Field(...)
    orders_filled: int = Field(...)
    orders_rejected: int = Field(...)

    # Timing
    duration_seconds: float = Field(...)

    # Hashes for replay verification
    target_weights_hash: str = Field(...)
    orders_hash: str = Field(...)


# =============================================================================
# Portfolio Models
# =============================================================================

class Position(BaseRecord):
    """A single position in the portfolio."""

    symbol: str = Field(...)
    shares: int = Field(...)
    avg_cost: float = Field(...)
    market_value: float = Field(...)
    weight: float = Field(...)
    unrealized_pnl: float = Field(...)
    unrealized_pnl_pct: float = Field(...)


class TargetWeight(BaseRecord):
    """Target weight for portfolio construction."""

    symbol: str = Field(...)
    target_weight: float = Field(..., ge=0, le=1)
    current_weight: float = Field(default=0.0, ge=0, le=1)
    score_final: float = Field(...)
    score_core: float = Field(...)
    penalty: float = Field(default=0.0)
    bonus: float = Field(default=0.0)
    position_cap: float = Field(default=0.05)
    sector: Optional[str] = Field(None)

    # Flags that affected this
    active_flags: List[str] = Field(default_factory=list)
    delay_trade: bool = Field(False)


class Order(BaseRecord):
    """An order to be executed."""

    order_id: str = Field(...)
    idempotency_key: str = Field(..., description="Key to prevent duplicate execution")
    run_id: str = Field(...)

    symbol: str = Field(...)
    side: str = Field(..., description="BUY or SELL")
    quantity: int = Field(...)
    order_type: str = Field(default="LIMIT")
    limit_price: Optional[float] = Field(None)

    # Status
    status: str = Field(default="PENDING")
    filled_quantity: int = Field(default=0)
    avg_fill_price: Optional[float] = Field(None)

    # Timing
    created_at: datetime = Field(default_factory=datetime.utcnow)
    submitted_at: Optional[datetime] = Field(None)
    filled_at: Optional[datetime] = Field(None)


class ReconcileResult(BaseRecord):
    """Result of position reconciliation."""

    reconcile_id: str = Field(...)
    asof_time: datetime = Field(...)

    # Comparison
    positions_matched: int = Field(...)
    positions_mismatched: int = Field(...)

    # Details
    mismatches: List[Dict[str, Any]] = Field(default_factory=list)

    # Action taken
    action: str = Field(...)  # PASS, WARN, FREEZE
    is_clean: bool = Field(...)
