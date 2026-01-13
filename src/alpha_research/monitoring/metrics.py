"""
Metrics Tracker for Alpha Research Trading System.

Tracks performance, risk, execution, data quality, and LLM metrics.
"""

import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from alpha_research.utils.config import load_config


class MetricsTracker:
    """
    Tracks and calculates system metrics.

    Key metrics:
    - Performance (returns, Sharpe, drawdown)
    - Risk (volatility, VAR)
    - Execution (fill rate, slippage)
    - Data quality (completeness, latency)
    - LLM (success rate, drift)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the metrics tracker.

        Args:
            config: Optional configuration override
        """
        if config is None:
            config = load_config('monitoring')

        self.config = config

        # Storage
        self._metrics_dir = Path('artifacts/metrics')
        self._metrics_dir.mkdir(parents=True, exist_ok=True)

        # Time series data
        self._nav_history: List[Tuple[date, float]] = []
        self._return_history: List[Tuple[date, float]] = []

        # Daily metrics
        self._daily_metrics: Dict[date, Dict[str, Any]] = {}

        # Execution metrics
        self._execution_metrics = defaultdict(list)

        # LLM metrics
        self._llm_metrics = defaultdict(list)

    def record_nav(self, nav: float, as_of: Optional[date] = None) -> None:
        """
        Record NAV for performance tracking.

        Args:
            nav: Net asset value
            as_of: Date (default: today)
        """
        if as_of is None:
            as_of = datetime.now().date()

        self._nav_history.append((as_of, nav))

        # Calculate return if we have previous NAV
        if len(self._nav_history) >= 2:
            prev_nav = self._nav_history[-2][1]
            if prev_nav > 0:
                ret = (nav - prev_nav) / prev_nav
                self._return_history.append((as_of, ret))

    def record_execution(
        self,
        symbol: str,
        side: str,
        quantity: int,
        expected_price: float,
        fill_price: float,
        fill_quantity: int,
        success: bool,
        as_of: Optional[date] = None,
    ) -> None:
        """
        Record execution metrics.

        Args:
            symbol: Stock symbol
            side: BUY or SELL
            quantity: Requested quantity
            expected_price: Expected execution price
            fill_price: Actual fill price
            fill_quantity: Filled quantity
            success: Whether execution succeeded
            as_of: Date
        """
        if as_of is None:
            as_of = datetime.now().date()

        slippage_bps = 0
        if expected_price > 0:
            slippage_bps = abs(fill_price - expected_price) / expected_price * 10000

        self._execution_metrics[as_of].append({
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'fill_quantity': fill_quantity,
            'fill_rate': fill_quantity / quantity if quantity > 0 else 0,
            'slippage_bps': slippage_bps,
            'success': success,
        })

    def record_llm_call(
        self,
        module_name: str,
        success: bool,
        latency_ms: float,
        tokens_used: int,
        schema_valid: bool,
        has_citations: bool,
        as_of: Optional[date] = None,
    ) -> None:
        """
        Record LLM call metrics.

        Args:
            module_name: Name of LLM module
            success: Whether call succeeded
            latency_ms: Latency in milliseconds
            tokens_used: Tokens consumed
            schema_valid: Whether output schema was valid
            has_citations: Whether output had citations
            as_of: Date
        """
        if as_of is None:
            as_of = datetime.now().date()

        self._llm_metrics[as_of].append({
            'module': module_name,
            'success': success,
            'latency_ms': latency_ms,
            'tokens_used': tokens_used,
            'schema_valid': schema_valid,
            'has_citations': has_citations,
        })

    def get_performance_metrics(
        self,
        lookback_days: int = 252,
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            lookback_days: Number of days for calculations

        Returns:
            Dictionary with performance metrics
        """
        if len(self._return_history) < 2:
            return {}

        # Get recent returns
        cutoff = datetime.now().date() - timedelta(days=lookback_days)
        recent_returns = [r for d, r in self._return_history if d >= cutoff]

        if not recent_returns:
            return {}

        returns = np.array(recent_returns)

        # Calculate metrics
        total_return = np.prod(1 + returns) - 1
        avg_return = np.mean(returns)
        volatility = np.std(returns) * np.sqrt(252)

        # Sharpe ratio (assuming 0 risk-free rate for simplicity)
        sharpe = avg_return * 252 / volatility if volatility > 0 else 0

        # Sortino ratio
        negative_returns = returns[returns < 0]
        downside_vol = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0
        sortino = avg_return * 252 / downside_vol if downside_vol > 0 else 0

        # Max drawdown
        nav_values = [n for _, n in self._nav_history if True]
        if nav_values:
            running_max = np.maximum.accumulate(nav_values)
            drawdowns = (running_max - nav_values) / running_max
            max_drawdown = np.max(drawdowns)
        else:
            max_drawdown = 0

        # Calmar ratio
        calmar = (avg_return * 252) / max_drawdown if max_drawdown > 0 else 0

        return {
            'total_return': total_return,
            'avg_daily_return': avg_return,
            'volatility_annual': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'days_tracked': len(returns),
        }

    def get_execution_metrics(
        self,
        lookback_days: int = 30,
    ) -> Dict[str, float]:
        """
        Calculate execution metrics.

        Args:
            lookback_days: Number of days for calculations

        Returns:
            Dictionary with execution metrics
        """
        cutoff = datetime.now().date() - timedelta(days=lookback_days)
        recent_executions = []

        for d, execs in self._execution_metrics.items():
            if d >= cutoff:
                recent_executions.extend(execs)

        if not recent_executions:
            return {}

        total = len(recent_executions)
        successful = sum(1 for e in recent_executions if e['success'])
        slippages = [e['slippage_bps'] for e in recent_executions if e['success']]
        fill_rates = [e['fill_rate'] for e in recent_executions]

        return {
            'total_orders': total,
            'successful_orders': successful,
            'success_rate': successful / total if total > 0 else 0,
            'avg_slippage_bps': np.mean(slippages) if slippages else 0,
            'median_slippage_bps': np.median(slippages) if slippages else 0,
            'max_slippage_bps': max(slippages) if slippages else 0,
            'avg_fill_rate': np.mean(fill_rates) if fill_rates else 0,
        }

    def get_llm_metrics(
        self,
        lookback_days: int = 7,
    ) -> Dict[str, Any]:
        """
        Calculate LLM metrics.

        Args:
            lookback_days: Number of days for calculations

        Returns:
            Dictionary with LLM metrics
        """
        cutoff = datetime.now().date() - timedelta(days=lookback_days)
        recent_calls = []

        for d, calls in self._llm_metrics.items():
            if d >= cutoff:
                recent_calls.extend(calls)

        if not recent_calls:
            return {}

        total = len(recent_calls)
        successful = sum(1 for c in recent_calls if c['success'])
        schema_valid = sum(1 for c in recent_calls if c['schema_valid'])
        with_citations = sum(1 for c in recent_calls if c['has_citations'])

        latencies = [c['latency_ms'] for c in recent_calls]
        tokens = [c['tokens_used'] for c in recent_calls]

        # By module
        module_stats = defaultdict(lambda: {'calls': 0, 'success': 0})
        for call in recent_calls:
            module = call['module']
            module_stats[module]['calls'] += 1
            if call['success']:
                module_stats[module]['success'] += 1

        return {
            'total_calls': total,
            'success_rate': successful / total if total > 0 else 0,
            'schema_valid_rate': schema_valid / total if total > 0 else 0,
            'citation_rate': with_citations / total if total > 0 else 0,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
            'total_tokens': sum(tokens),
            'avg_tokens_per_call': np.mean(tokens) if tokens else 0,
            'by_module': dict(module_stats),
        }

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics for dashboard display.

        Returns:
            Dictionary with all dashboard metrics
        """
        return {
            'performance': self.get_performance_metrics(),
            'execution': self.get_execution_metrics(),
            'llm': self.get_llm_metrics(),
            'nav_current': self._nav_history[-1][1] if self._nav_history else 0,
            'last_updated': datetime.now().isoformat(),
        }

    def save_daily_metrics(self, as_of: Optional[date] = None) -> None:
        """Save daily metrics to disk."""
        if as_of is None:
            as_of = datetime.now().date()

        metrics = self.get_dashboard_metrics()
        filepath = self._metrics_dir / f"metrics_{as_of.isoformat()}.json"

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

    def load_historical_metrics(
        self,
        days: int = 30,
    ) -> List[Dict[str, Any]]:
        """Load historical metrics from disk."""
        results = []

        for filepath in sorted(self._metrics_dir.glob("metrics_*.json"), reverse=True):
            if len(results) >= days:
                break

            try:
                with open(filepath, 'r') as f:
                    results.append(json.load(f))
            except:
                pass

        return results
