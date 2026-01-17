"""
Alternative Data Sources - 另类数据源

这是产生Alpha的关键: 独特数据 = 独特洞察

数据源分类:

1. 实时另类数据 (Real-time Alternative)
   - 国会议员交易 (Congress Trades)
   - 内部人交易 (Insider Trading)
   - 机构持仓 (13F Filings)
   - 卖空数据 (Short Interest)

2. 情绪/社交数据 (Sentiment/Social)
   - Reddit WSB 讨论
   - Twitter/X 金融情绪
   - 新闻情绪

3. 公司活动数据 (Corporate Activity)
   - 招聘数据 (Job Postings)
   - 专利申请 (Patent Filings)
   - 政府合同 (Government Contracts)
   - 游说活动 (Lobbying)

4. 经济指标 (Economic Indicators)
   - 供应链数据
   - 消费者数据

提供商:
- Finnhub.io: 综合金融数据 + 另类数据
- Polygon.io: 实时市场数据
- Quiver Quantitative: 国会交易/游说/合同
- Nasdaq Data Link (前Quandl): 机构级数据
"""

from .finnhub_client import FinnhubClient
from .polygon_client import PolygonClient
from .quiver_client import QuiverClient
from .aggregator import AlternativeDataAggregator

__all__ = [
    "FinnhubClient",
    "PolygonClient",
    "QuiverClient",
    "AlternativeDataAggregator",
]
