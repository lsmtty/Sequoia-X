"""知行砖型图红绿柱翻转选股策略。

对应通达信公式（修正版_知行趋势与砖型图组合选股）：

    砖型图指标：
        VAR1A = (HHV(HIGH,4) - CLOSE) / (HHV(HIGH,4) - LLV(LOW,4)) * 100 - 90
        VAR2A = SMA(VAR1A, 4, 1) + 100
        VAR3A = (CLOSE - LLV(LOW,4)) / (HHV(HIGH,4) - LLV(LOW,4)) * 100
        VAR4A = SMA(VAR3A, 6, 1)
        VAR5A = SMA(VAR4A, 6, 1) + 100
        VAR6A = VAR5A - VAR2A
        砖型图 = IF(VAR6A > 4, VAR6A - 4, 0)

    红柱：今日砖型图 > 昨日砖型图
    绿柱：今日砖型图 < 昨日砖型图

    选股条件：
        条件1：知行短期趋势线 > 知行多空线
        条件2：收盘价 > 知行多空线
        条件3：今日红柱（今日砖型图 > 昨日砖型图）
        条件4：昨日绿柱（昨日砖型图 < 前日砖型图）
        条件5：今日矩形实体高度 > 昨日矩形实体高度 × 2/3
"""

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


def _sma_td(series: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 SMA 算法：SMA(X, N, M) = (M*X + (N-M)*prev_SMA) / N。

    与 pandas rolling mean 不同，这是加权递推均线，需要逐步递推。
    """
    result = np.zeros(len(series))
    values = series.values
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = (m * values[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


def _calc_brick(df: pd.DataFrame) -> pd.Series:
    """计算知行砖型图序列。

    公式还原：
        VAR1A = (HHV4_high - close) / (HHV4_high - LLV4_low) * 100 - 90
        VAR2A = SMA(VAR1A, 4, 1) + 100
        VAR3A = (close - LLV4_low) / (HHV4_high - LLV4_low) * 100
        VAR4A = SMA(VAR3A, 6, 1)
        VAR5A = SMA(VAR4A, 6, 1) + 100
        VAR6A = VAR5A - VAR2A
        brick = max(VAR6A - 4, 0)
    """
    hhv4 = df["high"].rolling(4).max()
    llv4 = df["low"].rolling(4).min()
    denom = (hhv4 - llv4).replace(0, np.nan)

    var1a = (hhv4 - df["close"]) / denom * 100 - 90
    var1a = var1a.fillna(0)

    var2a = _sma_td(var1a, 4, 1) + 100

    var3a = (df["close"] - llv4) / denom * 100
    var3a = var3a.fillna(50)

    var4a = _sma_td(var3a, 6, 1)
    var5a = _sma_td(var4a, 6, 1) + 100

    var6a = var5a - var2a
    brick = var6a.clip(lower=4) - 4   # IF(VAR6A>4, VAR6A-4, 0)
    brick = brick.where(var6a > 4, 0)

    return brick


class ZhixingBrickReversalStrategy(BaseStrategy):
    """知行砖型图红绿柱翻转选股策略。

    选股条件（全部向量化，严禁 iterrows）：
    1. 知行短期趋势线 > 知行多空线（上升趋势环境）
    2. 收盘价 > 知行多空线（价格在多空线之上）
    3. 今日砖型图 > 昨日砖型图（今日红柱）
    4. 昨日砖型图 < 前日砖型图（昨日绿柱，即昨日是回调）
    5. 今日矩形实体高度 > 昨日矩形实体高度 × 2/3（今日反弹力度够强）

    指标定义：
    - 知行短期趋势线 = EMA(EMA(C,10), 10)
    - 知行多空线     = (MA14 + MA28 + MA57 + MA114) / 4
    - 砖型图         = 通达信原始公式还原（见模块文档）

    Attributes:
        webhook_key: 路由到 'zhixing_brick' 专属飞书机器人。
    """

    webhook_key: str = "zhixing_brick"
    _MIN_BARS: int = 115  # MA114 最低需求

    def run(self) -> list[str]:
        """遍历全市场，返回满足知行砖型图翻转条件的股票代码列表。"""
        symbols = self.engine.get_local_symbols()
        selected: list[str] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                # ── 知行趋势线 ──
                ema10 = df["close"].ewm(span=10, adjust=False).mean()
                trend_line = ema10.ewm(span=10, adjust=False).mean()     # 知行短期趋势线
                bull_bear_line = (
                    df["close"].rolling(14).mean()
                    + df["close"].rolling(28).mean()
                    + df["close"].rolling(57).mean()
                    + df["close"].rolling(114).mean()
                ) / 4                                                     # 知行多空线

                # ── 砖型图 ──
                brick = _calc_brick(df)

                # ── 取最后三根 K 线（今日、昨日、前日） ──
                b_today  = brick.iloc[-1]   # 今日砖型图
                b_prev1  = brick.iloc[-2]   # 昨日砖型图
                b_prev2  = brick.iloc[-3]   # 前日砖型图

                close_today   = df["close"].iloc[-1]
                trend_today   = trend_line.iloc[-1]
                bull_bear_today = bull_bear_line.iloc[-1]

                if any(pd.isna(x) for x in [b_today, b_prev1, b_prev2, trend_today, bull_bear_today]):
                    continue

                # 条件1：趋势线在多空线之上
                cond1 = trend_today > bull_bear_today

                # 条件2：收盘价在多空线之上
                cond2 = close_today > bull_bear_today

                # 条件3：今日红柱（砖型图升高）
                cond3 = b_today > b_prev1

                # 条件4：昨日绿柱（昨日砖型图比前日低）
                cond4 = b_prev1 < b_prev2

                # 条件5：今日实体高度 > 昨日实体高度 × 2/3
                today_height  = abs(b_today - b_prev1)
                yest_height   = abs(b_prev1 - b_prev2)
                cond5 = today_height > yest_height * (2 / 3)

                if cond1 and cond2 and cond3 and cond4 and cond5:
                    selected.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] ZhixingBrickReversalStrategy 计算失败：{exc}")
                continue

        logger.info(f"ZhixingBrickReversalStrategy 选出 {len(selected)} 只股票")
        return selected
