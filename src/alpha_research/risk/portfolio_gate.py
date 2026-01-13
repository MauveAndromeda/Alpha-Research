"""
Portfolio Gate for Alpha Research Trading System.

Enforces portfolio construction constraints.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from alpha_research.data.models import TargetWeight, Order
from alpha_research.utils.config import load_config


@dataclass
class GateResult:
    """Result of portfolio gate evaluation."""
    approved: bool
    target_weights: List[TargetWeight]
    rejected_trades: List[Dict[str, Any]] = field(default_factory=list)
    adjustments: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class PortfolioGate:
    """
    Portfolio gate that enforces construction constraints.

    Key responsibilities:
    1. Enforce holdings count limits
    2. Apply position and sector caps
    3. Check turnover limits
    4. Validate liquidity requirements
    5. Filter dust trades
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the portfolio gate.

        Args:
            config: Optional configuration override
        """
        if config is None:
            gov_config = load_config('governance_policy')
            config = gov_config.get('portfolio_gate', {})

        self.config = config

        # Holdings constraints
        holdings = config.get('holdings', {})
        self.target_holdings = holdings.get('target', 25)
        self.min_holdings = holdings.get('min', 20)
        self.max_holdings = holdings.get('max', 30)

        # Cap constraints
        caps = config.get('caps', {})
        self.base_position_cap = caps.get('base_position_cap', 0.05)
        self.sector_cap = caps.get('sector_cap', 0.25)

        # Turnover constraints
        turnover = config.get('turnover', {})
        self.monthly_turnover_cap = turnover.get('monthly_turnover_cap', 0.40)

        # Liquidity constraints
        liquidity = config.get('liquidity', {})
        self.min_price = liquidity.get('min_price', 5.0)
        self.min_adv_dollar = liquidity.get('min_adv_dollar_60d', 50000000)
        self.min_adv_multiple = liquidity.get('min_adv_multiple', 30)

        # Trade filter
        trade_filter = config.get('trade_filter', {})
        self.min_trade_notional = trade_filter.get('min_trade_notional', 300)

    def evaluate(
        self,
        proposed_weights: List[TargetWeight],
        current_weights: Dict[str, float],
        market_data: pd.DataFrame,
        total_capital: float,
        month_turnover_used: float = 0.0,
    ) -> GateResult:
        """
        Evaluate proposed portfolio against constraints.

        Args:
            proposed_weights: List of proposed target weights
            current_weights: Current portfolio weights
            market_data: Market data with prices and ADV
            total_capital: Total portfolio capital
            month_turnover_used: Turnover already used this month

        Returns:
            GateResult with approved weights and adjustments
        """
        adjustments = []
        rejected = []
        warnings = []

        # Convert to working format
        weights_df = pd.DataFrame([w.dict() for w in proposed_weights])

        # Step 1: Check holdings count
        if len(weights_df) > self.max_holdings:
            # Keep only top by target_weight
            weights_df = weights_df.nlargest(self.max_holdings, 'target_weight')
            adjustments.append({
                'type': 'holdings_trimmed',
                'from': len(proposed_weights),
                'to': self.max_holdings,
            })

        if len(weights_df) < self.min_holdings:
            warnings.append(
                f"Only {len(weights_df)} holdings, below minimum {self.min_holdings}"
            )

        # Step 2: Apply position caps
        for idx, row in weights_df.iterrows():
            # Use the more restrictive of base cap and symbol-specific cap
            effective_cap = min(self.base_position_cap, row.get('position_cap', 1.0))

            if row['target_weight'] > effective_cap:
                adjustments.append({
                    'type': 'position_capped',
                    'symbol': row['symbol'],
                    'from': row['target_weight'],
                    'to': effective_cap,
                })
                weights_df.at[idx, 'target_weight'] = effective_cap

        # Step 3: Apply sector caps
        weights_df = self._apply_sector_caps(weights_df, adjustments)

        # Step 4: Check liquidity
        weights_df, liquidity_rejected = self._check_liquidity(
            weights_df, market_data, total_capital
        )
        rejected.extend(liquidity_rejected)

        # Step 5: Check turnover
        turnover_remaining = self.monthly_turnover_cap - month_turnover_used
        weights_df, turnover_adjusted = self._check_turnover(
            weights_df, current_weights, turnover_remaining
        )
        if turnover_adjusted:
            adjustments.append({
                'type': 'turnover_constrained',
                'remaining_budget': turnover_remaining,
            })

        # Step 6: Filter dust trades
        weights_df, dust_rejected = self._filter_dust_trades(
            weights_df, current_weights, total_capital
        )
        rejected.extend(dust_rejected)

        # Step 7: Re-normalize weights
        total_weight = weights_df['target_weight'].sum()
        if total_weight > 0 and total_weight != 1.0:
            weights_df['target_weight'] = weights_df['target_weight'] / total_weight
            adjustments.append({
                'type': 'renormalized',
                'from_total': total_weight,
                'to_total': 1.0,
            })

        # Convert back to TargetWeight objects
        approved_weights = []
        for _, row in weights_df.iterrows():
            # Find original and update
            for orig in proposed_weights:
                if orig.symbol == row['symbol']:
                    updated = TargetWeight(
                        symbol=row['symbol'],
                        target_weight=row['target_weight'],
                        current_weight=current_weights.get(row['symbol'], 0),
                        score_final=orig.score_final,
                        score_core=orig.score_core,
                        penalty=orig.penalty,
                        bonus=orig.bonus,
                        position_cap=row.get('position_cap', self.base_position_cap),
                        sector=orig.sector,
                        active_flags=orig.active_flags,
                        delay_trade=orig.delay_trade,
                    )
                    approved_weights.append(updated)
                    break

        return GateResult(
            approved=True,
            target_weights=approved_weights,
            rejected_trades=rejected,
            adjustments=adjustments,
            warnings=warnings,
        )

    def _apply_sector_caps(
        self,
        weights_df: pd.DataFrame,
        adjustments: List[Dict],
    ) -> pd.DataFrame:
        """Apply sector caps and redistribute weight."""
        if 'sector' not in weights_df.columns:
            return weights_df

        result = weights_df.copy()

        for sector in result['sector'].dropna().unique():
            sector_mask = result['sector'] == sector
            sector_total = result.loc[sector_mask, 'target_weight'].sum()

            if sector_total > self.sector_cap:
                # Scale down sector weights
                scale = self.sector_cap / sector_total
                result.loc[sector_mask, 'target_weight'] *= scale

                adjustments.append({
                    'type': 'sector_capped',
                    'sector': sector,
                    'from': sector_total,
                    'to': self.sector_cap,
                })

        return result

    def _check_liquidity(
        self,
        weights_df: pd.DataFrame,
        market_data: pd.DataFrame,
        total_capital: float,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Check liquidity requirements."""
        rejected = []

        # Build ADV map
        adv_map = {}
        price_map = {}
        for symbol in weights_df['symbol']:
            symbol_data = market_data[market_data['symbol'] == symbol]
            if len(symbol_data) > 0:
                latest = symbol_data.iloc[-1]
                adv_map[symbol] = latest.get('adv_dollar_60d', float('inf'))
                price_map[symbol] = latest.get('close', 0)

        # Filter by ADV
        keep_mask = []
        for _, row in weights_df.iterrows():
            symbol = row['symbol']
            adv = adv_map.get(symbol, 0)
            price = price_map.get(symbol, 0)

            # Check minimum ADV
            if adv < self.min_adv_dollar:
                rejected.append({
                    'symbol': symbol,
                    'reason': f'ADV ${adv:,.0f} below minimum ${self.min_adv_dollar:,.0f}',
                })
                keep_mask.append(False)
                continue

            # Check ADV multiple
            position_value = row['target_weight'] * total_capital
            if adv > 0 and position_value > 0:
                adv_multiple = adv / position_value
                if adv_multiple < self.min_adv_multiple:
                    rejected.append({
                        'symbol': symbol,
                        'reason': f'Position too large relative to ADV ({adv_multiple:.1f}x)',
                    })
                    keep_mask.append(False)
                    continue

            # Check price
            if price < self.min_price:
                rejected.append({
                    'symbol': symbol,
                    'reason': f'Price ${price:.2f} below minimum ${self.min_price:.2f}',
                })
                keep_mask.append(False)
                continue

            keep_mask.append(True)

        return weights_df[keep_mask].copy(), rejected

    def _check_turnover(
        self,
        weights_df: pd.DataFrame,
        current_weights: Dict[str, float],
        remaining_budget: float,
    ) -> Tuple[pd.DataFrame, bool]:
        """Check and constrain turnover."""
        result = weights_df.copy()

        # Calculate proposed turnover
        all_symbols = set(result['symbol']) | set(current_weights.keys())

        turnover = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target_row = result[result['symbol'] == symbol]
            target = target_row['target_weight'].iloc[0] if len(target_row) > 0 else 0
            turnover += abs(target - current)

        turnover = turnover / 2  # One-way

        if turnover <= remaining_budget:
            return result, False

        # Need to constrain - blend with current
        blend_factor = remaining_budget / turnover

        for idx, row in result.iterrows():
            symbol = row['symbol']
            current = current_weights.get(symbol, 0)
            target = row['target_weight']
            result.at[idx, 'target_weight'] = (
                blend_factor * target + (1 - blend_factor) * current
            )

        return result, True

    def _filter_dust_trades(
        self,
        weights_df: pd.DataFrame,
        current_weights: Dict[str, float],
        total_capital: float,
    ) -> Tuple[pd.DataFrame, List[Dict]]:
        """Filter out dust trades (below minimum notional)."""
        rejected = []
        keep_mask = []

        for _, row in weights_df.iterrows():
            symbol = row['symbol']
            current = current_weights.get(symbol, 0)
            target = row['target_weight']

            trade_value = abs(target - current) * total_capital

            if trade_value < self.min_trade_notional and trade_value > 0:
                # Keep current weight instead
                rejected.append({
                    'symbol': symbol,
                    'reason': f'Trade ${trade_value:.2f} below minimum ${self.min_trade_notional:.2f}',
                    'action': 'kept_current_weight',
                })
                # Modify to keep current
                keep_mask.append(True)  # Keep but don't trade
            else:
                keep_mask.append(True)

        return weights_df[keep_mask].copy(), rejected

    def generate_orders(
        self,
        approved_weights: List[TargetWeight],
        current_positions: Dict[str, int],
        prices: Dict[str, float],
        total_capital: float,
        run_id: str,
    ) -> List[Order]:
        """
        Generate orders from approved weights.

        Args:
            approved_weights: List of approved target weights
            current_positions: Current positions (shares)
            prices: Current prices
            total_capital: Total portfolio capital
            run_id: Current run ID

        Returns:
            List of Order objects
        """
        from alpha_research.utils.hashing import generate_order_idempotency_key

        orders = []

        for weight in approved_weights:
            symbol = weight.symbol
            price = prices.get(symbol, 0)

            if price <= 0:
                continue

            # Calculate target shares
            target_value = weight.target_weight * total_capital
            target_shares = int(target_value / price)

            # Get current shares
            current_shares = current_positions.get(symbol, 0)

            # Calculate trade
            trade_shares = target_shares - current_shares

            if trade_shares == 0:
                continue

            # Determine side
            side = "BUY" if trade_shares > 0 else "SELL"
            quantity = abs(trade_shares)

            # Generate idempotency key
            idempotency_key = generate_order_idempotency_key(
                run_id, symbol, side, quantity
            )

            order = Order(
                order_id=f"{run_id}_{symbol}_{side}",
                idempotency_key=idempotency_key,
                run_id=run_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=price,  # Will be adjusted by execution
            )

            orders.append(order)

        return orders
