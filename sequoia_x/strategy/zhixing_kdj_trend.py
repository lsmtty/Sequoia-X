"""知行KDJ低位+趋势选股策略。

对应通达信公式：
    知行短期趋势线 = EMA(EMA(C,10),10)
    知行多空线     = (MA14 + MA28 + MA57 + MA114) / 4
    KDJ: N=9, M1=3, M2=3（标准通达信算法）
    涨幅条件: -2.5% < 今日涨幅 < 3%
    振幅条件: (H-L)/L < 7%
    趋势条件: 知行短期趋势线 > 知行多空线
    信号: J <= 13 AND 涨幅条件 AND 振幅条件 AND 趋势条件
"""

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger
from sequoia_x.strategy.base import BaseStrategy

logger = get_logger(__name__)


def _sma_td(series: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 SMA 算法：SMA(X, N, M) = (M*X + (N-M)*prev_SMA) / N。

    与 pandas rolling mean 不同，这是加权递推均线。
    """
    result = np.zeros(len(series))
    values = series.values
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = (m * values[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


class ZhixingKdjTrendStrategy(BaseStrategy):
    """知行KDJ低位+趋势选股策略。

    选股条件（全部向量化，严禁 iterrows）：
    1. J 值 <= 13（KDJ 超卖低位）
    2. 今日涨幅在 -2.5% ~ +3% 之间（排除大涨大跌异常日）
    3. 今日振幅 (H-L)/L < 7%（排除异常跳空/波动日）
    4. 知行短期趋势线 > 知行多空线（上升趋势中捕捉回调低点）

    指标定义：
    - 知行短期趋势线 = EMA(EMA(C,10), 10)
    - 知行多空线     = (MA14 + MA28 + MA57 + MA114) / 4
    - KDJ: N=9, M1=3, M2=3（通达信标准 SMA 递推算法）

    Attributes:
        webhook_key: 路由到 'zhixing_kdj' 专属飞书机器人。
    """

    webhook_key: str = "zhixing_kdj"
    _MIN_BARS: int = 115  # 至少需要 115 根 K 线（MA114 需要）

    def run(self) -> list[str]:
        """遍历全市场，返回满足知行KDJ低位+趋势条件的股票代码列表。"""
        symbols = self.engine.get_local_symbols()
        selected: list[str] = []

        for symbol in symbols:
            try:
                df = self.engine.get_ohlcv(symbol)
                if len(df) < self._MIN_BARS:
                    continue

                # ── 知行趋势线 ──
                ema10 = df["close"].ewm(span=10, adjust=False).mean()
                trend_line = ema10.ewm(span=10, adjust=False).mean()   # 知行短期趋势线
                bull_bear_line = (
                    df["close"].rolling(14).mean()
                    + df["close"].rolling(28).mean()
                    + df["close"].rolling(57).mean()
                    + df["close"].rolling(114).mean()
                ) / 4                                                   # 知行多空线

                # ── KDJ（通达信标准算法） ──
                n = 9
                llv = df["low"].rolling(n).min()
                hhv = df["high"].rolling(n).max()
                denom = hhv - llv
                rsv = ((df["close"] - llv) / denom.replace(0, np.nan) * 100).fillna(50)

                k = _sma_td(rsv, 3, 1)
                d = _sma_td(k,   3, 1)
                j = 3 * k - 2 * d

                # ── 取最后一根 K 线 ──
                last_idx = -1
                close_today  = df["close"].iloc[last_idx]
                close_prev   = df["close"].iloc[-2]
                high_today   = df["high"].iloc[last_idx]
                low_today    = df["low"].iloc[last_idx]

                j_val         = j.iloc[last_idx]
                trend_val     = trend_line.iloc[last_idx]
                bull_bear_val = bull_bear_line.iloc[last_idx]

                if pd.isna(j_val) or pd.isna(trend_val) or pd.isna(bull_bear_val):
                    continue

                # 条件 1：J 值超卖
                cond_j = j_val <= 13

                # 条件 2：涨幅 -2.5% ~ +3%
                pct = (close_today / close_prev - 1) * 100
                cond_pct = -2.5 < pct < 3.0

                # 条件 3：振幅 < 7%
                if low_today == 0:
                    continue
                amplitude = (high_today - low_today) / low_today * 100
                cond_amp = amplitude < 7.0

                # 条件 4：短期趋势线在多空线之上
                cond_trend = trend_val > bull_bear_val

                if cond_j and cond_pct and cond_amp and cond_trend:
                    selected.append(symbol)

            except Exception as exc:
                logger.warning(f"[{symbol}] ZhixingKdjTrendStrategy 计算失败：{exc}")
                continue

        logger.info(f"ZhixingKdjTrendStrategy 选出 {len(selected)} 只股票")
        return selected
