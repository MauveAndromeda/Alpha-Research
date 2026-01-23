"""
Reconciler for Alpha Research Trading System.

Compares local positions with broker positions to detect mismatches.
Triggers trading freeze on mismatch.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from alpha_research.data.models import ReconcileResult
from alpha_research.utils.enums import ErrorCode
from alpha_research.utils.hashing import compute_hash
from alpha_research.utils.config import load_config


@dataclass
class PositionMismatch:
    """Details of a position mismatch."""
    symbol: str
    local_shares: int
    broker_shares: int
    difference: int
    local_value: float
    broker_value: float
    value_difference: float
    severity: str  # LOW, MEDIUM, HIGH


class Reconciler:
    """
    Reconciles local positions with broker positions.

    Key responsibilities:
    1. Compare position counts
    2. Identify mismatches
    3. Trigger freeze on significant differences
    4. Generate reconciliation reports
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the reconciler.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('execution_policy')

        self.config = config

        recon_config = config.get('reconciliation', {})
        self.enabled = recon_config.get('enabled', True)

        tolerance = recon_config.get('tolerance', {})
        self.shares_tolerance = tolerance.get('shares', 0)
        self.value_pct_tolerance = tolerance.get('value_pct', 0.01)

        self.on_mismatch = recon_config.get('on_mismatch', 'FREEZE_TRADING')

        # Storage
        self._reconcile_dir = Path(
            config.get('paths', {}).get('reconcile_dir', 'artifacts/reconcile')
        )
        self._reconcile_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._is_frozen = False
        self._last_reconcile: Optional[ReconcileResult] = None

    def reconcile(
        self,
        local_positions: Dict[str, int],
        broker_positions: Dict[str, int],
        prices: Dict[str, float],
        asof_time: Optional[datetime] = None,
    ) -> ReconcileResult:
        """
        Perform reconciliation.

        Args:
            local_positions: Local position tracking (symbol -> shares)
            broker_positions: Broker reported positions
            prices: Current prices for value calculation
            asof_time: As-of timestamp

        Returns:
            ReconcileResult
        """
        if asof_time is None:
            asof_time = datetime.utcnow()

        mismatches = []
        matched = 0

        # All symbols in either
        all_symbols = set(local_positions.keys()) | set(broker_positions.keys())

        for symbol in all_symbols:
            local = local_positions.get(symbol, 0)
            broker = broker_positions.get(symbol, 0)
            price = prices.get(symbol, 0)

            difference = local - broker

            # Check if within tolerance
            if abs(difference) <= self.shares_tolerance:
                matched += 1
                continue

            # Calculate value difference
            local_value = local * price
            broker_value = broker * price
            value_diff = local_value - broker_value

            # Determine severity
            if abs(difference) > 100 or abs(value_diff) > 10000:
                severity = 'HIGH'
            elif abs(difference) > 10 or abs(value_diff) > 1000:
                severity = 'MEDIUM'
            else:
                severity = 'LOW'

            mismatch = PositionMismatch(
                symbol=symbol,
                local_shares=local,
                broker_shares=broker,
                difference=difference,
                local_value=local_value,
                broker_value=broker_value,
                value_difference=value_diff,
                severity=severity,
            )
            mismatches.append(mismatch)

        # Determine action
        is_clean = len(mismatches) == 0
        action = 'PASS' if is_clean else self.on_mismatch

        # Create result
        result = ReconcileResult(
            reconcile_id=compute_hash(f"reconcile_{asof_time.isoformat()}")[:16],
            asof_time=asof_time,
            positions_matched=matched,
            positions_mismatched=len(mismatches),
            mismatches=[{
                'symbol': m.symbol,
                'local_shares': m.local_shares,
                'broker_shares': m.broker_shares,
                'difference': m.difference,
                'value_difference': m.value_difference,
                'severity': m.severity,
            } for m in mismatches],
            action=action,
            is_clean=is_clean,
        )

        # Update state
        self._last_reconcile = result
        if not is_clean and action == 'FREEZE_TRADING':
            self._is_frozen = True

        # Save result
        self._save_result(result)

        return result

    def is_frozen(self) -> bool:
        """Check if trading is frozen due to reconciliation failure."""
        return self._is_frozen

    def unfreeze(self, reason: str = "Manual unfreeze") -> None:
        """
        Manually unfreeze trading.

        Args:
            reason: Reason for unfreezing
        """
        self._is_frozen = False
        print(f"Trading unfrozen: {reason}")

    def get_last_result(self) -> Optional[ReconcileResult]:
        """Get the last reconciliation result."""
        return self._last_reconcile

    def _save_result(self, result: ReconcileResult) -> None:
        """Save reconciliation result to disk."""
        date_str = result.asof_time.strftime("%Y%m%d_%H%M%S")
        filepath = self._reconcile_dir / f"reconcile_{date_str}.json"

        with open(filepath, 'w') as f:
            json.dump(result.model_dump(), f, indent=2, default=str)

    def load_history(
        self,
        days: int = 30,
    ) -> List[ReconcileResult]:
        """
        Load reconciliation history.

        Args:
            days: Number of days of history to load

        Returns:
            List of ReconcileResult objects
        """
        results = []

        for filepath in sorted(self._reconcile_dir.glob("reconcile_*.json"), reverse=True):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                results.append(ReconcileResult(**data))

                if len(results) >= days:
                    break
            except Exception as e:
                print(f"Error loading reconcile file {filepath}: {e}")

        return results

    def get_summary_stats(
        self,
        history: List[ReconcileResult],
    ) -> Dict[str, Any]:
        """
        Get summary statistics from reconciliation history.

        Args:
            history: List of reconciliation results

        Returns:
            Dictionary with summary stats
        """
        if not history:
            return {}

        clean_count = sum(1 for r in history if r.is_clean)
        total_mismatches = sum(r.positions_mismatched for r in history)

        # Find most common mismatch symbols
        symbol_counts = {}
        for result in history:
            for mismatch in result.mismatches:
                symbol = mismatch.get('symbol')
                symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1

        return {
            'total_reconciliations': len(history),
            'clean_reconciliations': clean_count,
            'clean_rate': clean_count / len(history) if history else 0,
            'total_mismatches': total_mismatches,
            'avg_mismatches_per_day': total_mismatches / len(history) if history else 0,
            'most_common_mismatches': sorted(
                symbol_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
        }


class PositionTracker:
    """
    Tracks local positions based on executed orders.

    Used to compare against broker positions.
    """

    def __init__(self):
        self._positions: Dict[str, int] = {}
        self._cost_basis: Dict[str, float] = {}

    def update_from_fill(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
    ) -> None:
        """
        Update positions from a fill.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Filled quantity
            price: Fill price
        """
        current = self._positions.get(symbol, 0)
        current_cost = self._cost_basis.get(symbol, 0)

        if side == "BUY":
            # Update cost basis (weighted average)
            total_value = current * current_cost + quantity * price
            new_position = current + quantity
            if new_position > 0:
                self._cost_basis[symbol] = total_value / new_position
            self._positions[symbol] = new_position

        else:  # SELL
            self._positions[symbol] = current - quantity
            # Keep cost basis unchanged for remaining shares

        # Clean up zero positions
        if self._positions.get(symbol, 0) == 0:
            self._positions.pop(symbol, None)
            self._cost_basis.pop(symbol, None)

    def get_positions(self) -> Dict[str, int]:
        """Get current positions."""
        return self._positions.copy()

    def get_cost_basis(self) -> Dict[str, float]:
        """Get cost basis for positions."""
        return self._cost_basis.copy()

    def set_positions(
        self,
        positions: Dict[str, int],
        cost_basis: Optional[Dict[str, float]] = None,
    ) -> None:
        """
        Set positions directly (e.g., from broker sync).

        Args:
            positions: Symbol -> shares mapping
            cost_basis: Symbol -> avg cost mapping
        """
        self._positions = positions.copy()
        if cost_basis:
            self._cost_basis = cost_basis.copy()

    def calculate_pnl(
        self,
        prices: Dict[str, float],
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate P&L for all positions.

        Args:
            prices: Current prices

        Returns:
            Dictionary with P&L details per symbol
        """
        pnl = {}

        for symbol, shares in self._positions.items():
            price = prices.get(symbol, 0)
            cost = self._cost_basis.get(symbol, 0)

            market_value = shares * price
            cost_value = shares * cost
            unrealized = market_value - cost_value
            unrealized_pct = (unrealized / cost_value) if cost_value > 0 else 0

            pnl[symbol] = {
                'shares': shares,
                'cost_basis': cost,
                'current_price': price,
                'market_value': market_value,
                'cost_value': cost_value,
                'unrealized_pnl': unrealized,
                'unrealized_pnl_pct': unrealized_pct,
            }

        return pnl
