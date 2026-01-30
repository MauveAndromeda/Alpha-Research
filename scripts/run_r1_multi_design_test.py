#!/usr/bin/env python3
"""
DEEPSEEK R1 MULTI-DESIGN TEST
==============================

尝试多种R1集成设计，寻找最高alpha的方案:

设计1: 市场择时专家 - 只调整仓位，不选股
设计2: 逆向信号专家 - 极端恐慌时加仓
设计3: 质量过滤专家 - 只剔除最差的，不调整排名
设计4: 趋势确认专家 - 只增强最强信号
设计5: 行业轮动专家 - 选择最佳行业
设计6: 止盈止损专家 - 动态调整止损位
设计7: 动量加速专家 - 识别加速中的动量
"""

import hashlib
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

try:
    from numba import jit
except ImportError:
    def jit(*a, **k):
        def d(f): return f
        return d

UNIVERSE = [
    'AAPL', 'MSFT', 'NVDA', 'ADBE', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU',
    'INTC', 'CSCO', 'ORCL', 'IBM', 'ADI', 'MCHP', 'AMD', 'AVGO',
    'UNH', 'LLY', 'AMGN', 'GILD', 'MDT', 'SYK', 'ABT', 'JNJ', 'MRK', 'PFE',
    'HD', 'LOW', 'COST', 'NKE', 'SBUX', 'MCD', 'TJX', 'ROST',
    'JPM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
    'CAT', 'DE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX', 'GE', 'BA',
    'PG', 'KO', 'PEP', 'WMT', 'CL',
    'XOM', 'CVX', 'COP', 'SLB',
]

SECTOR_MAP = {s: 'Tech' for s in ['AAPL', 'MSFT', 'NVDA', 'ADBE', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC', 'MU', 'INTC', 'CSCO', 'ORCL', 'IBM', 'ADI', 'MCHP', 'AMD', 'AVGO']}
SECTOR_MAP.update({s: 'Health' for s in ['UNH', 'LLY', 'AMGN', 'GILD', 'MDT', 'SYK', 'ABT', 'JNJ', 'MRK', 'PFE']})
SECTOR_MAP.update({s: 'Consumer' for s in ['HD', 'LOW', 'COST', 'NKE', 'SBUX', 'MCD', 'TJX', 'ROST']})
SECTOR_MAP.update({s: 'Finance' for s in ['JPM', 'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'V', 'MA']})
SECTOR_MAP.update({s: 'Industrial' for s in ['CAT', 'DE', 'HON', 'UNP', 'UPS', 'LMT', 'RTX', 'GE', 'BA']})
SECTOR_MAP.update({s: 'Staples' for s in ['PG', 'KO', 'PEP', 'WMT', 'CL']})
SECTOR_MAP.update({s: 'Energy' for s in ['XOM', 'CVX', 'COP', 'SLB']})

SECTORS = ['Tech', 'Health', 'Consumer', 'Finance', 'Industrial', 'Staples', 'Energy']


def fetch_data(symbols, start, end):
    cache_dir = Path.home() / ".alpha_research" / "cache_r1_multi"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.md5(f"{sorted(symbols)}_{start}_{end}".encode()).hexdigest()[:12]
    cache_file = cache_dir / f"data_{cache_key}.parquet"
    
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    
    import yfinance as yf
    print(f"Fetching {len(symbols)} symbols...")
    results = []
    
    with ThreadPoolExecutor(max_workers=15) as executor:
        def fetch_one(sym):
            try:
                hist = yf.Ticker(sym).history(start=start - timedelta(days=500), end=end, auto_adjust=True)
                if len(hist) >= 50:
                    hist = hist.reset_index()
                    hist['symbol'] = sym
                    hist['trade_date'] = hist['Date'].dt.date
                    return hist[['symbol', 'trade_date', 'Close']].rename(columns={'Close': 'close'})
            except:
                pass
            return None
        
        futures = list(executor.map(fetch_one, symbols))
        for r in futures:
            if r is not None:
                results.append(r)
    
    df = pd.concat(results, ignore_index=True)
    df.to_parquet(cache_file)
    return df


def build_matrix(df, symbols, dates):
    sym_to_idx = {s: i for i, s in enumerate(symbols)}
    date_to_idx = {d: i for i, d in enumerate(dates)}
    prices = np.full((len(symbols), len(dates)), np.nan)
    
    for _, row in df.iterrows():
        s, d = row['symbol'], row['trade_date']
        if s in sym_to_idx and d in date_to_idx:
            prices[sym_to_idx[s], date_to_idx[d]] = row['close']
    
    for i in range(len(symbols)):
        last = np.nan
        for j in range(len(dates)):
            if np.isnan(prices[i, j]):
                prices[i, j] = last
            else:
                last = prices[i, j]
    
    return prices, sym_to_idx, date_to_idx


@jit(nopython=True, fastmath=True, cache=True)
def calc_return(p, idx, n):
    if idx < n or p[idx-n] <= 0:
        return np.nan
    return p[idx] / p[idx-n] - 1

@jit(nopython=True, fastmath=True, cache=True)
def calc_vol(p, idx, n=21):
    if idx < n:
        return 0.20
    rets = np.zeros(n - 1)
    for i in range(n - 1):
        if p[idx-n+i+1] > 0 and p[idx-n+i] > 0:
            rets[i] = p[idx-n+i+1] / p[idx-n+i] - 1
    return np.std(rets) * np.sqrt(252)

@jit(nopython=True, fastmath=True, cache=True)
def calc_sma(p, idx, n):
    if idx < n:
        return np.nan
    return np.mean(p[idx-n+1:idx+1])


# ============================================================
# R1 设计模式 (模拟R1的决策逻辑)
# ============================================================

class R1Design:
    """R1设计基类"""
    def __init__(self, name):
        self.name = name
    
    def get_exposure(self, spy, vix, idx):
        """市场仓位调整"""
        return 1.0
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        """调整个股评分"""
        return base_score
    
    def get_stop_loss(self, sym_idx, prices, idx, default_stop):
        """动态止损"""
        return default_stop
    
    def filter_candidates(self, candidates, prices, idx):
        """过滤候选股"""
        return candidates


class Design0_PureMomentum(R1Design):
    """设计0: 纯动量基准"""
    def __init__(self):
        super().__init__("0_Pure_Momentum")


class Design1_MarketTimer(R1Design):
    """
    设计1: 市场择时专家
    - 不干预选股
    - 只根据市场状态调整整体仓位
    - VIX高 + SPY下跌 = 恐慌反弹机会 → 加仓
    - VIX低 + SPY高估 = 风险 → 减仓
    """
    def __init__(self):
        super().__init__("1_Market_Timer")
    
    def get_exposure(self, spy, vix, idx):
        vix_now = vix[idx] if idx < len(vix) else 20
        
        spy_sma200 = calc_sma(spy, idx, 200)
        spy_above_200 = spy[idx] > spy_sma200 if not np.isnan(spy_sma200) else True
        
        spy_ret_1m = calc_return(spy, idx, 21)
        
        # 计算VIX百分位
        vix_lookback = vix[max(0, idx-252):idx+1]
        vix_pct = sum(1 for v in vix_lookback if v < vix_now) / len(vix_lookback) if len(vix_lookback) > 0 else 0.5
        
        # 极端恐慌后的逆向机会
        if vix_now > 35 and vix_pct > 0.90:
            if spy_ret_1m is not None and not np.isnan(spy_ret_1m) and spy_ret_1m < -0.10:
                return 1.20  # 加仓20%
            return 1.0
        
        # 熊市
        if not spy_above_200:
            return 0.80  # 减仓20%
        
        # 极端自满
        if vix_now < 12:
            return 0.90  # 减仓10%
        
        return 1.0


class Design2_ContrarianSignal(R1Design):
    """
    设计2: 逆向信号专家
    - 市场恐慌时大幅加仓
    - 寻找超跌反弹机会
    """
    def __init__(self):
        super().__init__("2_Contrarian")
    
    def get_exposure(self, spy, vix, idx):
        vix_now = vix[idx] if idx < len(vix) else 20
        spy_ret_1m = calc_return(spy, idx, 21)
        spy_ret_3m = calc_return(spy, idx, 63)
        
        # 极端恐慌 = 最大机会
        if vix_now > 40:
            return 1.30  # 加仓30%
        elif vix_now > 30:
            return 1.15
        
        # 急跌后反弹
        if spy_ret_1m is not None and not np.isnan(spy_ret_1m):
            if spy_ret_1m < -0.15:
                return 1.25
            elif spy_ret_1m < -0.08:
                return 1.10
        
        return 1.0
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        p = prices[sym_idx]
        
        # 寻找超跌股
        ret_1m = calc_return(p, idx, 21)
        ret_3m = calc_return(p, idx, 63)
        
        if np.isnan(ret_1m):
            return base_score
        
        # 3个月涨但近1月跌 = 回调买入机会
        if not np.isnan(ret_3m) and ret_3m > 0.10 and ret_1m < -0.10:
            return base_score * 1.20
        
        return base_score


class Design3_QualityFilter(R1Design):
    """
    设计3: 质量过滤专家
    - 只剔除明显的"差"股票
    - 不调整好股票的排名
    """
    def __init__(self):
        super().__init__("3_Quality_Filter")
    
    def filter_candidates(self, candidates, prices, idx):
        filtered = []
        
        for c in candidates:
            sym_idx = c['sym_idx']
            p = prices[sym_idx]
            
            # 检查是否有问题
            vol = calc_vol(p, idx)
            ret_1m = calc_return(p, idx, 21)
            
            # 剔除: 极端波动
            if vol > 0.60:
                continue
            
            # 剔除: 动量崩溃
            if not np.isnan(ret_1m) and ret_1m < -0.25:
                continue
            
            filtered.append(c)
        
        return filtered


class Design4_TrendConfirm(R1Design):
    """
    设计4: 趋势确认专家
    - 只增强最强趋势的股票
    - 多重均线确认
    """
    def __init__(self):
        super().__init__("4_Trend_Confirm")
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        p = prices[sym_idx]
        
        sma20 = calc_sma(p, idx, 20)
        sma50 = calc_sma(p, idx, 50)
        sma200 = calc_sma(p, idx, 200)
        
        if np.isnan(sma20) or np.isnan(sma50) or np.isnan(sma200):
            return base_score
        
        # 完美多头排列: 价格 > SMA20 > SMA50 > SMA200
        if p[idx] > sma20 > sma50 > sma200:
            return base_score * 1.25  # 加强25%
        
        # 趋势向上但未完美
        if p[idx] > sma200 and sma50 > sma200:
            return base_score * 1.10
        
        return base_score


class Design5_SectorRotation(R1Design):
    """
    设计5: 行业轮动专家
    - 计算各行业动量
    - 倾向于最强行业
    """
    def __init__(self):
        super().__init__("5_Sector_Rotation")
        self.sector_strength = {}
    
    def _calc_sector_strength(self, candidates, prices, idx):
        """计算行业强度"""
        sector_scores = defaultdict(list)
        
        for c in candidates:
            sector = c['sector']
            sector_scores[sector].append(c['mom_score'])
        
        self.sector_strength = {}
        for sector, scores in sector_scores.items():
            self.sector_strength[sector] = np.mean(scores) if scores else 0
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        if not self.sector_strength:
            self._calc_sector_strength(all_candidates, prices, idx)
        
        # 行业排名
        sorted_sectors = sorted(self.sector_strength.items(), key=lambda x: x[1], reverse=True)
        sector_rank = next((i for i, (s, _) in enumerate(sorted_sectors) if s == sector), 3)
        
        # 前2行业加分
        if sector_rank == 0:
            return base_score * 1.20
        elif sector_rank == 1:
            return base_score * 1.10
        elif sector_rank >= 5:
            return base_score * 0.90
        
        return base_score


class Design6_DynamicStop(R1Design):
    """
    设计6: 动态止损专家
    - 牛市用宽止损
    - 熊市用紧止损
    """
    def __init__(self):
        super().__init__("6_Dynamic_Stop")
    
    def get_stop_loss(self, spy, vix, idx, default_stop=0.10):
        vix_now = vix[idx] if idx < len(vix) else 20
        
        spy_sma200 = calc_sma(spy, idx, 200)
        spy_above_200 = spy[idx] > spy_sma200 if not np.isnan(spy_sma200) else True
        
        # 牛市 + 低VIX: 宽止损
        if spy_above_200 and vix_now < 20:
            return 0.15  # 15%
        
        # 熊市: 紧止损
        if not spy_above_200:
            return 0.08  # 8%
        
        # 高VIX: 中等止损
        if vix_now > 25:
            return 0.12
        
        return default_stop


class Design7_MomentumAccel(R1Design):
    """
    设计7: 动量加速专家
    - 识别动量正在加速的股票
    - 动量加速 = 更高权重
    """
    def __init__(self):
        super().__init__("7_Momentum_Accel")
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        p = prices[sym_idx]
        
        mom_12m = calc_return(p, idx, 252)
        mom_6m = calc_return(p, idx, 126)
        mom_3m = calc_return(p, idx, 63)
        mom_1m = calc_return(p, idx, 21)
        
        if np.isnan(mom_12m) or np.isnan(mom_6m) or np.isnan(mom_3m):
            return base_score
        
        # 动量加速: 近期 > 远期
        accel_6_12 = mom_6m - (mom_12m / 2)
        accel_3_6 = mom_3m - (mom_6m / 2)
        
        # 强加速
        if accel_6_12 > 0.05 and accel_3_6 > 0.03:
            return base_score * 1.30
        
        # 中等加速
        if accel_6_12 > 0 and accel_3_6 > 0:
            return base_score * 1.15
        
        # 减速
        if accel_3_6 < -0.05:
            return base_score * 0.85
        
        return base_score


class Design8_Combined(R1Design):
    """
    设计8: 综合设计
    - 结合最有效的元素:
      - 市场择时 (Design1)
      - 动量加速 (Design7)
      - 趋势确认 (Design4)
    """
    def __init__(self):
        super().__init__("8_Combined_Best")
        self.timer = Design1_MarketTimer()
        self.accel = Design7_MomentumAccel()
        self.trend = Design4_TrendConfirm()
    
    def get_exposure(self, spy, vix, idx):
        return self.timer.get_exposure(spy, vix, idx)
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        # 先应用动量加速
        score = self.accel.adjust_score(sym_idx, prices, idx, base_score, sector, all_candidates)
        # 再应用趋势确认
        score = self.trend.adjust_score(sym_idx, prices, idx, score, sector, all_candidates)
        return score


class Design9_Aggressive(R1Design):
    """
    设计9: 激进设计
    - 集中持仓 (3只)
    - 强逆向
    - 强加速
    """
    def __init__(self):
        super().__init__("9_Aggressive")
    
    def get_exposure(self, spy, vix, idx):
        vix_now = vix[idx] if idx < len(vix) else 20
        spy_ret_1m = calc_return(spy, idx, 21)
        
        # 极端逆向
        if vix_now > 35:
            return 1.40  # 加仓40%
        
        if spy_ret_1m is not None and not np.isnan(spy_ret_1m) and spy_ret_1m < -0.12:
            return 1.30
        
        return 1.0
    
    def adjust_score(self, sym_idx, prices, idx, base_score, sector, all_candidates):
        p = prices[sym_idx]
        
        mom_6m = calc_return(p, idx, 126)
        mom_3m = calc_return(p, idx, 63)
        
        if np.isnan(mom_6m) or np.isnan(mom_3m):
            return base_score
        
        # 强加速bonus
        accel = mom_3m - (mom_6m / 2)
        if accel > 0.08:
            return base_score * 1.40
        elif accel > 0.04:
            return base_score * 1.20
        
        return base_score


# ============================================================
# STRATEGY ENGINE
# ============================================================

class StrategyEngine:
    """策略引擎"""
    
    def __init__(self, prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx,
                 r1_design, n_holdings=5):
        self.prices = prices
        self.spy = spy
        self.vix = vix
        self.symbols = symbols
        self.dates = dates
        self.sym_to_idx = sym_to_idx
        self.date_to_idx = date_to_idx
        self.r1_design = r1_design
        self.n_holdings = n_holdings
    
    def run(self, start, end):
        dates = [d for d in self.dates if start <= d <= end]
        
        cash = 100000.0
        positions = {}
        peaks = {}
        nav_history = []
        high_water = cash
        
        for i, current_date in enumerate(dates):
            idx = self.date_to_idx[current_date]
            
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p):
                    nav += shares * p
            
            # 动态止损
            if hasattr(self.r1_design, 'get_stop_loss'):
                stop_loss = self.r1_design.get_stop_loss(self.spy, self.vix, idx, 0.10)
            else:
                stop_loss = 0.10
            
            for sym_idx, shares in list(positions.items()):
                p = self.prices[sym_idx, idx]
                if np.isnan(p):
                    continue
                peaks[sym_idx] = max(peaks.get(sym_idx, p), p)
                if (peaks[sym_idx] - p) / peaks[sym_idx] > stop_loss:
                    cash += shares * p * 0.998
                    del positions[sym_idx]
                    del peaks[sym_idx]
            
            # 周度再平衡
            if current_date.weekday() == 4:
                # 计算候选股
                candidates = []
                
                for sym_idx in range(len(self.symbols)):
                    p = self.prices[sym_idx]
                    
                    mom_12m = calc_return(p, idx-1, 252)
                    mom_6m = calc_return(p, idx-1, 126)
                    
                    if np.isnan(mom_12m) or mom_12m < 0:
                        continue
                    
                    sma200 = calc_sma(p, idx-1, 200)
                    if not np.isnan(sma200) and p[idx-1] < sma200:
                        continue
                    
                    # 基础动量评分
                    mom_12_1 = calc_return(p, idx-22, 231) if idx >= 253 else mom_12m
                    mom_6_1 = calc_return(p, idx-22, 105) if idx >= 148 else mom_6m
                    mom_score = mom_12_1 * 0.60 + mom_6_1 * 0.40
                    
                    vol = calc_vol(p, idx-1)
                    
                    candidates.append({
                        'sym_idx': sym_idx,
                        'symbol': self.symbols[sym_idx],
                        'sector': SECTOR_MAP.get(self.symbols[sym_idx], 'Other'),
                        'mom_score': mom_score,
                        'vol': vol,
                        'price': p[idx],
                    })
                
                # R1过滤
                if hasattr(self.r1_design, 'filter_candidates'):
                    candidates = self.r1_design.filter_candidates(candidates, self.prices, idx-1)
                
                # R1调整评分
                for c in candidates:
                    adjusted = self.r1_design.adjust_score(
                        c['sym_idx'], self.prices, idx-1, 
                        c['mom_score'], c['sector'], candidates
                    )
                    c['final_score'] = adjusted
                
                # 排序
                candidates.sort(key=lambda x: x['final_score'], reverse=True)
                
                # 行业分散选择
                selected = []
                sector_counts = defaultdict(int)
                for c in candidates:
                    if sector_counts[c['sector']] < 2:
                        selected.append(c)
                        sector_counts[c['sector']] += 1
                    if len(selected) >= self.n_holdings:
                        break
                
                # R1仓位调整
                exposure = self.r1_design.get_exposure(self.spy, self.vix, idx)
                
                if selected:
                    inv_vols = [1.0/max(0.15, s['vol']) for s in selected]
                    total = sum(inv_vols)
                    
                    nav = cash
                    for sym_idx, shares in positions.items():
                        p = self.prices[sym_idx, idx]
                        if not np.isnan(p):
                            nav += shares * p
                    
                    effective_nav = nav * exposure
                    
                    targets = {}
                    for j, s in enumerate(selected):
                        weight = inv_vols[j] / total
                        alloc = effective_nav * weight * 0.98
                        shares = int(alloc / s['price'])
                        if shares > 0:
                            targets[s['sym_idx']] = shares
                    
                    for sym_idx in list(positions.keys()):
                        if sym_idx not in targets:
                            p = self.prices[sym_idx, idx]
                            if not np.isnan(p):
                                cash += positions[sym_idx] * p * 0.998
                            del positions[sym_idx]
                            if sym_idx in peaks:
                                del peaks[sym_idx]
                    
                    for sym_idx, target in targets.items():
                        current = positions.get(sym_idx, 0)
                        delta = target - current
                        p = self.prices[sym_idx, idx]
                        if delta > 0:
                            cost = delta * p * 1.002
                            if cost <= cash:
                                cash -= cost
                                positions[sym_idx] = current + delta
                                if sym_idx not in peaks:
                                    peaks[sym_idx] = p
                        elif delta < 0:
                            cash += abs(delta) * p * 0.998
                            positions[sym_idx] = current + delta
                            if positions[sym_idx] <= 0:
                                del positions[sym_idx]
            
            nav = cash
            for sym_idx, shares in positions.items():
                p = self.prices[sym_idx, idx]
                if not np.isnan(p):
                    nav += shares * p
            
            high_water = max(high_water, nav)
            dd = (high_water - nav) / high_water
            nav_history.append({'date': current_date, 'nav': nav, 'dd': dd})
        
        navs = np.array([h['nav'] for h in nav_history])
        rets = np.diff(navs) / navs[:-1]
        
        total_ret = (navs[-1] - 100000) / 100000
        n_years = (end - start).days / 365.25
        ann_ret = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0
        ann_vol = np.std(rets) * np.sqrt(252)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else 0
        max_dd = max(h['dd'] for h in nav_history)
        
        return {
            'ann_return': ann_ret,
            'sharpe': sharpe,
            'max_dd': max_dd,
        }


def main():
    print("=" * 80)
    print("DEEPSEEK R1 MULTI-DESIGN TEST")
    print("=" * 80)
    print("""
测试多种R1设计，寻找最高alpha方案:

  设计0: 纯动量 (基准)
  设计1: 市场择时 - VIX/趋势调整仓位
  设计2: 逆向信号 - 恐慌加仓
  设计3: 质量过滤 - 剔除差股票
  设计4: 趋势确认 - 多均线确认
  设计5: 行业轮动 - 选择强势行业
  设计6: 动态止损 - 牛熊止损调整
  设计7: 动量加速 - 识别加速动量
  设计8: 综合最佳 - 1+4+7组合
  设计9: 激进设计 - 集中+逆向+加速
    """)
    
    start_date = date(2005, 1, 1)
    end_date = date(2025, 12, 31)
    
    print("Loading data...")
    df = fetch_data(UNIVERSE + ['SPY'], start_date, end_date)
    symbols = [s for s in df['symbol'].unique() if s in UNIVERSE]
    dates = sorted(df['trade_date'].unique())
    
    prices, sym_to_idx, date_to_idx = build_matrix(df, symbols, dates)
    
    spy = np.zeros(len(dates))
    for _, row in df[df['symbol'] == 'SPY'].iterrows():
        if row['trade_date'] in date_to_idx:
            spy[date_to_idx[row['trade_date']]] = row['close']
    last = np.nan
    for i in range(len(spy)):
        if spy[i] <= 0: spy[i] = last
        else: last = spy[i]
    
    try:
        import yfinance as yf
        vix_hist = yf.Ticker("^VIX").history(start=start_date - timedelta(days=100), end=end_date)
        vix_dict = {row.name.date(): row['Close'] for _, row in vix_hist.iterrows()}
        vix = np.array([vix_dict.get(d, 20) for d in dates])
    except:
        vix = np.full(len(dates), 20.0)
    
    print(f"Loaded {len(symbols)} stocks")
    
    # 所有设计
    designs = [
        (Design0_PureMomentum(), 5),
        (Design1_MarketTimer(), 5),
        (Design2_ContrarianSignal(), 5),
        (Design3_QualityFilter(), 5),
        (Design4_TrendConfirm(), 5),
        (Design5_SectorRotation(), 5),
        (Design6_DynamicStop(), 5),
        (Design7_MomentumAccel(), 5),
        (Design8_Combined(), 5),
        (Design9_Aggressive(), 3),  # 激进用3只
    ]
    
    # ========================================
    # 5年测试
    # ========================================
    print("\n" + "=" * 80)
    print("5-YEAR TEST (2020-2025)")
    print("=" * 80)
    
    test_start = date(2020, 1, 1)
    test_end = dates[-1]
    
    print(f"\n{'Design':<30} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'vs Base':>10}")
    print("-" * 70)
    
    results_5y = []
    baseline_5y = None
    
    for design, n_hold in designs:
        engine = StrategyEngine(
            prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx,
            r1_design=design,
            n_holdings=n_hold,
        )
        r = engine.run(test_start, test_end)
        r['name'] = design.name
        results_5y.append(r)
        
        if baseline_5y is None:
            baseline_5y = r['ann_return']
            vs_base = ""
        else:
            vs_base = f"{r['ann_return'] - baseline_5y:>+9.1%}"
        
        flag = "★" if r['ann_return'] >= baseline_5y + 0.02 else " "
        print(f"{flag}{design.name:<29} {r['ann_return']:>9.1%} {r['sharpe']:>8.2f} {r['max_dd']:>7.1%} {vs_base}")
    
    # ========================================
    # 20年测试
    # ========================================
    print("\n" + "=" * 80)
    print("20-YEAR TEST (2006-2025)")
    print("=" * 80)
    
    test_start_20 = date(2006, 1, 1)
    
    print(f"\n{'Design':<30} {'Ann Ret':>10} {'Sharpe':>8} {'MaxDD':>8} {'vs Base':>10}")
    print("-" * 70)
    
    results_20y = []
    baseline_20y = None
    
    for design, n_hold in designs:
        engine = StrategyEngine(
            prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx,
            r1_design=design,
            n_holdings=n_hold,
        )
        r = engine.run(test_start_20, test_end)
        r['name'] = design.name
        results_20y.append(r)
        
        if baseline_20y is None:
            baseline_20y = r['ann_return']
            vs_base = ""
        else:
            vs_base = f"{r['ann_return'] - baseline_20y:>+9.1%}"
        
        flag = "★" if r['ann_return'] >= baseline_20y + 0.02 else " "
        print(f"{flag}{design.name:<29} {r['ann_return']:>9.1%} {r['sharpe']:>8.2f} {r['max_dd']:>7.1%} {vs_base}")
    
    # ========================================
    # 找到最佳设计
    # ========================================
    print("\n" + "=" * 80)
    print("TOP PERFORMERS")
    print("=" * 80)
    
    # 按5年收益排序
    sorted_5y = sorted(results_5y, key=lambda x: x['ann_return'], reverse=True)
    sorted_20y = sorted(results_20y, key=lambda x: x['ann_return'], reverse=True)
    
    print("\n5年最佳:")
    for i, r in enumerate(sorted_5y[:3]):
        print(f"  {i+1}. {r['name']}: {r['ann_return']:.1%} / Sharpe {r['sharpe']:.2f}")
    
    print("\n20年最佳:")
    for i, r in enumerate(sorted_20y[:3]):
        print(f"  {i+1}. {r['name']}: {r['ann_return']:.1%} / Sharpe {r['sharpe']:.2f}")
    
    # ========================================
    # Walk-Forward 最佳设计
    # ========================================
    best_design_name = sorted_5y[0]['name']
    best_design = next((d for d, _ in designs if d.name == best_design_name), designs[0][0])
    best_n_hold = next((n for d, n in designs if d.name == best_design_name), 5)
    
    print(f"\n" + "=" * 80)
    print(f"WALK-FORWARD: {best_design_name}")
    print("=" * 80)
    
    wf_periods = [
        ('2006-2010', date(2006, 1, 1), date(2010, 12, 31)),
        ('2011-2015', date(2011, 1, 1), date(2015, 12, 31)),
        ('2016-2020', date(2016, 1, 1), date(2020, 12, 31)),
        ('2021-2025', date(2021, 1, 1), date(2025, 12, 31)),
    ]
    
    print(f"\n{'Period':<12} {'Ann Ret':>10} {'Sharpe':>8}")
    print("-" * 35)
    
    wf_results = []
    for pname, p_start, p_end in wf_periods:
        engine = StrategyEngine(
            prices, spy, vix, symbols, dates, sym_to_idx, date_to_idx,
            r1_design=best_design,
            n_holdings=best_n_hold,
        )
        r = engine.run(p_start, p_end)
        wf_results.append(r)
        print(f"{pname:<12} {r['ann_return']:>9.1%} {r['sharpe']:>8.2f}")
    
    avg_ret = np.mean([r['ann_return'] for r in wf_results])
    avg_sharpe = np.mean([r['sharpe'] for r in wf_results])
    print(f"\n{'Average':<12} {avg_ret:>9.1%} {avg_sharpe:>8.2f}")
    
    # ========================================
    # SUMMARY
    # ========================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    best_5y = sorted_5y[0]
    best_20y = sorted_20y[0]
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  多设计测试结果                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  【5年最佳设计】                                                 │
│  {best_5y['name']:<60}│
│  年化收益: {best_5y['ann_return']:>6.1%}                                           │
│  Sharpe:   {best_5y['sharpe']:>6.2f}                                           │
│  vs基准:   {best_5y['ann_return'] - baseline_5y:>+5.1%}                                            │
│                                                                 │
│  【20年最佳设计】                                                │
│  {best_20y['name']:<60}│
│  年化收益: {best_20y['ann_return']:>6.1%}                                           │
│  Sharpe:   {best_20y['sharpe']:>6.2f}                                           │
│  vs基准:   {best_20y['ann_return'] - baseline_20y:>+5.1%}                                            │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  距离30%目标: {30 - best_5y['ann_return']*100:>+5.1f}% (5年)                              │
│  距离Sharpe 1.5: {1.5 - best_5y['sharpe']:>+5.2f} (5年)                              │
└─────────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 80)


if __name__ == "__main__":
    main()
