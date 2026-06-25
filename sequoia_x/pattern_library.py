"""历史形态库构建模块。

设计：
1. 数据来源
   - 主数据：本地 SQLite（2024-01-02 至今），只取近 300 个交易日
   - 熊市补充：通过 baostock 单独拉取指定熊市区间，存到独立 SQLite 表
     - 2015-07-01 ~ 2015-12-31（股灾后半段）
     - 2018-01-01 ~ 2018-12-31（中美贸易战）
     - 2022-01-01 ~ 2022-10-31（俄乌+美联储加息）
     - 2024-01-01 ~ 2024-02-29（雪球爆仓，已在主库）

2. 形态识别
   - 在每只股票的历史数据上，找出所有触发过基础选股信号的历史节点
   - 截取信号日前 30 根 K 线作为"形态窗口"（归一化）

3. 未来表现评分（超额收益，剔除牛市基准抬升）
   - 基准：沪深300（sh.000300）同期涨幅，缓存到 data/benchmark.db
   - 指标A：信号后 5 日内突破信号前30日最高价（0 or 1）
   - 指标B：信号后30日 个股超额收益（个股涨幅 - 大盘涨幅），超过+10%得满分
   - 指标C：信号后90日 个股超额收益，超过+20%得满分
   - 综合得分 = wA*A + wB*B + wC*C
   - 只保留综合得分 >= threshold 的形态进入形态库

4. 存储格式
   - 每条记录：symbol, signal_date, market_type(bull/bear),
               pattern_array(json 30×5 OHLCV), future_score, 各子项得分
   - 存入 data/pattern_library.db
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine

logger = get_logger(__name__)

# ── 熊市区间配置 ──────────────────────────────────────────────────────────

BEAR_PERIODS: list[tuple[str, str, str]] = [
    ("2015-07-01", "2015-12-31", "bear_2015"),
    ("2018-01-01", "2018-12-31", "bear_2018"),
    ("2022-01-01", "2022-10-31", "bear_2022"),
    # 2024-01-01~02-29 已在主库，通过 market_type 标记即可，无需重拉
]

# ── 未来表现权重（可在 .env 或外部覆盖）─────────────────────────────────

DEFAULT_WEIGHTS = {"w_5d_breakout": 0.3, "w_30d_gain": 0.4, "w_90d_gain": 0.3}
DEFAULT_SCORE_THRESHOLD = 0.0   # 0 = 全部入库（包含差形态），让匹配结果有区分度

PATTERN_WINDOW = 30             # 形态窗口：信号日前N根K线
MAIN_LOOKBACK_DAYS = 300        # 主数据只取近N个交易日

# ── 形态库 DDL ────────────────────────────────────────────────────────────

_CREATE_PATTERN_TABLE = """
CREATE TABLE IF NOT EXISTS patterns (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT    NOT NULL,
    signal_date  TEXT    NOT NULL,
    market_type  TEXT    NOT NULL,  -- 'bull' / 'bear_2015' / 'bear_2018' / 'bear_2022'
    pattern_json TEXT    NOT NULL,  -- 归一化后的 30×5 OHLCV 数组（JSON）
    score_5d     REAL,
    score_30d    REAL,
    score_90d    REAL,
    future_score REAL    NOT NULL,
    UNIQUE(symbol, signal_date)
);
"""
_CREATE_PATTERN_INDEX = """
CREATE INDEX IF NOT EXISTS idx_pattern_score
ON patterns (future_score DESC, market_type);
"""

# ── 辅助：baostock 指标计算 ───────────────────────────────────────────────

def _sma_td(series: pd.Series, n: int, m: int) -> pd.Series:
    result = np.zeros(len(series))
    v = series.values
    result[0] = v[0]
    for i in range(1, len(v)):
        result[i] = (m * v[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


def _has_kdj_signal(df: pd.DataFrame) -> pd.Series:
    """返回每日是否满足 KDJ 策略信号（布尔 Series）。"""
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    trend_line = ema10.ewm(span=10, adjust=False).mean()
    bull_bear = (
        df["close"].rolling(14).mean()
        + df["close"].rolling(28).mean()
        + df["close"].rolling(57).mean()
        + df["close"].rolling(114).mean()
    ) / 4

    llv = df["low"].rolling(9).min()
    hhv = df["high"].rolling(9).max()
    denom = (hhv - llv).replace(0, np.nan)
    rsv = ((df["close"] - llv) / denom * 100).fillna(50)
    k = _sma_td(rsv, 3, 1)
    d = _sma_td(k, 3, 1)
    j = 3 * k - 2 * d

    pct = df["close"].pct_change() * 100
    amp = (df["high"] - df["low"]) / df["low"].replace(0, np.nan) * 100

    cond_j     = j <= 13
    cond_pct   = (pct > -2.5) & (pct < 3.0)
    cond_amp   = amp < 7.0
    cond_trend = trend_line > bull_bear

    return cond_j & cond_pct & cond_amp & cond_trend


def _has_brick_signal(df: pd.DataFrame) -> pd.Series:
    """返回每日是否满足砖型图策略信号（布尔 Series）。"""
    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    trend_line = ema10.ewm(span=10, adjust=False).mean()
    bull_bear = (
        df["close"].rolling(14).mean()
        + df["close"].rolling(28).mean()
        + df["close"].rolling(57).mean()
        + df["close"].rolling(114).mean()
    ) / 4

    hhv4 = df["high"].rolling(4).max()
    llv4 = df["low"].rolling(4).min()
    denom4 = (hhv4 - llv4).replace(0, np.nan)
    var1a = (hhv4 - df["close"]) / denom4 * 100 - 90
    var1a = var1a.fillna(0)
    var2a = _sma_td(var1a, 4, 1) + 100
    var3a = ((df["close"] - llv4) / denom4 * 100).fillna(50)
    var4a = _sma_td(var3a, 6, 1)
    var5a = _sma_td(var4a, 6, 1) + 100
    var6a = var5a - var2a
    brick = pd.Series(np.where(var6a > 4, var6a - 4, 0), index=df.index)

    b_today = brick
    b_prev1 = brick.shift(1)
    b_prev2 = brick.shift(2)

    cond1 = trend_line > bull_bear
    cond2 = df["close"] > bull_bear
    cond3 = b_today > b_prev1
    cond4 = b_prev1 < b_prev2
    today_h = (b_today - b_prev1).abs()
    yest_h  = (b_prev1 - b_prev2).abs()
    cond5 = today_h > yest_h * (2 / 3)

    return cond1 & cond2 & cond3 & cond4 & cond5


def _any_signal(df: pd.DataFrame) -> pd.Series:
    """任意策略触发即为信号日。"""
    return _has_kdj_signal(df) | _has_brick_signal(df)


# ── 归一化形态 ────────────────────────────────────────────────────────────

def _normalize_pattern(window: pd.DataFrame) -> list[list[float]]:
    """把 30根K线 OHLCV 归一化到 [0,1]，基准为窗口内 close 的 min/max。

    返回 list[list[float]]，每行 [open, high, low, close, volume]。
    """
    base = window["close"].iloc[0]
    if base == 0:
        base = 1.0
    cols = ["open", "high", "low", "close"]
    arr = window[cols].copy()
    arr = arr / base  # 以第一日收盘价归一

    # volume 单独归一化
    v_max = window["volume"].max()
    v_min = window["volume"].min()
    v_range = v_max - v_min if v_max != v_min else 1.0
    vol_norm = ((window["volume"] - v_min) / v_range).tolist()

    result = []
    for i, row in enumerate(arr.itertuples()):
        result.append([
            round(row.open, 6), round(row.high, 6),
            round(row.low, 6),  round(row.close, 6),
            round(vol_norm[i], 6),
        ])
    return result


# ── 大盘基准缓存 ──────────────────────────────────────────────────────────

_BENCHMARK_CACHE: dict[str, float] | None = None   # {date_str: close}
_BENCHMARK_DB = "data/benchmark.db"
_BENCHMARK_CODE = "sh.000300"   # 沪深300


def _ensure_benchmark_loaded(start_date: str = "2014-01-01") -> dict[str, float]:
    """加载/缓存沪深300日收盘价。优先读本地 benchmark.db，否则通过 baostock 拉取。

    Returns:
        {date_str: close_price}
    """
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is not None:
        return _BENCHMARK_CACHE

    db_path = Path(_BENCHMARK_DB)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 先读本地 ──
    if db_path.exists():
        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT date, close FROM benchmark ORDER BY date"
            ).fetchall()
        if rows:
            _BENCHMARK_CACHE = {r[0]: float(r[1]) for r in rows}
            return _BENCHMARK_CACHE

    # ── 本地无数据，通过 baostock 拉取 ──
    logger.info(f"拉取大盘基准 {_BENCHMARK_CODE} ({start_date} ~ 今)...")
    import baostock as bs
    today = date.today().strftime("%Y-%m-%d")
    bs.login()
    try:
        rs = bs.query_history_k_data_plus(
            _BENCHMARK_CODE,
            "date,close",
            start_date=start_date,
            end_date=today,
            frequency="d",
            adjustflag="3",
        )
        rows_raw = []
        while rs.next():
            rows_raw.append(rs.get_row_data())
    finally:
        bs.logout()

    if not rows_raw:
        logger.warning("大盘基准数据为空，超额收益评分将退化为绝对涨幅")
        _BENCHMARK_CACHE = {}
        return _BENCHMARK_CACHE

    # 存本地
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS benchmark (date TEXT PRIMARY KEY, close REAL)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO benchmark VALUES (?,?)",
            [(r[0], float(r[1])) for r in rows_raw if r[1]],
        )
        conn.commit()

    _BENCHMARK_CACHE = {r[0]: float(r[1]) for r in rows_raw if r[1]}
    logger.info(f"大盘基准加载完成，共 {len(_BENCHMARK_CACHE)} 个交易日")
    return _BENCHMARK_CACHE


def _get_benchmark_return(
    bm: dict[str, float],
    dates: pd.Series,
    signal_date: str,
    n_days: int,
) -> float:
    """计算大盘在信号日后 n_days 个交易日内的涨幅（最高收盘 / 信号日收盘 - 1）。

    若日期不存在基准数据，返回 0.0。
    """
    if not bm or signal_date not in bm:
        return 0.0

    signal_close = bm[signal_date]
    if signal_close == 0:
        return 0.0

    # 找 dates 序列中 signal_date 之后的 n_days 个交易日
    date_list = sorted(bm.keys())
    try:
        idx = date_list.index(signal_date)
    except ValueError:
        return 0.0

    future_dates = date_list[idx + 1: idx + 1 + n_days]
    if not future_dates:
        return 0.0

    max_close = max(bm[d] for d in future_dates if d in bm)
    return (max_close - signal_close) / signal_close


# ── 未来收益评分（超额收益 Alpha）──────────────────────────────────────────

def _calc_future_score(
    df: pd.DataFrame,
    signal_idx: int,
    weights: dict[str, float],
    benchmark: dict[str, float] | None = None,
) -> tuple[float, float, float, float]:
    """计算信号日后的超额收益得分（个股涨幅 - 大盘涨幅 = Alpha）。

    评分体系：
    - 指标A（score_5d）：5日内突破信号前30日最高价（0 or 1）
    - 指标B（score_30d）：30日超额收益 >= +10% 得满分，线性映射 [-5%, +10%] → [0, 1]
    - 指标C（score_90d）：90日超额收益 >= +20% 得满分，线性映射 [-5%, +20%] → [0, 1]

    Returns:
        (score_5d, score_30d, score_90d, future_score)
    """
    base_close = df["close"].iloc[signal_idx]
    if base_close == 0:
        return 0.0, 0.0, 0.0, 0.0

    signal_date = df["date"].iloc[signal_idx] if "date" in df.columns else None

    # 加载大盘基准（懒加载）
    if benchmark is None:
        try:
            benchmark = _ensure_benchmark_loaded()
        except Exception:
            benchmark = {}

    # 信号前30日最高价（用于5日突破判断）
    pre_high = df["high"].iloc[max(0, signal_idx - 30): signal_idx].max()

    future = df.iloc[signal_idx + 1:]

    # A：5日内突破30日最高价
    score_5d = 0.0
    if len(future) >= 1:
        w5 = future.head(5)
        if not w5.empty and not pd.isna(pre_high) and pre_high > 0:
            score_5d = 1.0 if w5["high"].max() > pre_high else 0.0

    # B：30日超额收益 Alpha
    score_30d = 0.0
    if len(future) >= 5:
        w30 = future.head(30)
        stock_gain_30 = (w30["high"].max() - base_close) / base_close

        bm_gain_30 = 0.0
        if signal_date and benchmark:
            bm_gain_30 = _get_benchmark_return(benchmark, df["date"] if "date" in df.columns else pd.Series(), signal_date, 30)

        alpha_30 = stock_gain_30 - bm_gain_30
        # [-5%, +10%] → [0, 1]，低于-5%得0，超过+10%得满分
        score_30d = min(1.0, max(0.0, (alpha_30 - (-0.05)) / (0.10 - (-0.05))))

    # C：90日超额收益 Alpha
    score_90d = 0.0
    if len(future) >= 20:
        w90 = future.head(90)
        stock_gain_90 = (w90["high"].max() - base_close) / base_close

        bm_gain_90 = 0.0
        if signal_date and benchmark:
            bm_gain_90 = _get_benchmark_return(benchmark, df["date"] if "date" in df.columns else pd.Series(), signal_date, 90)

        alpha_90 = stock_gain_90 - bm_gain_90
        # [-5%, +20%] → [0, 1]，低于-5%得0，超过+20%得满分
        score_90d = min(1.0, max(0.0, (alpha_90 - (-0.05)) / (0.20 - (-0.05))))

    future_score = (
        weights["w_5d_breakout"] * score_5d
        + weights["w_30d_gain"]  * score_30d
        + weights["w_90d_gain"]  * score_90d
    )
    return round(score_5d, 4), round(score_30d, 4), round(score_90d, 4), round(future_score, 4)


# ── 形态库主类 ────────────────────────────────────────────────────────────

class PatternLibrary:
    """历史形态库，负责构建、存储和查询优质历史形态。"""

    def __init__(
        self,
        settings: Settings,
        lib_db_path: str = "data/pattern_library.db",
        weights: dict[str, float] | None = None,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.main_engine = DataEngine(settings)
        self.lib_db_path = lib_db_path
        self.weights = weights or DEFAULT_WEIGHTS
        self.score_threshold = score_threshold
        self._init_lib_db()

    def _init_lib_db(self) -> None:
        Path(self.lib_db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.lib_db_path) as conn:
            conn.execute(_CREATE_PATTERN_TABLE)
            conn.execute(_CREATE_PATTERN_INDEX)
            conn.commit()
        logger.info(f"形态库初始化完成：{self.lib_db_path}")

    # ── 从主库读取主数据（近300交易日）────────────────────────────────────

    def _load_main_data(self, symbol: str) -> pd.DataFrame:
        """从主 SQLite 读取近 MAIN_LOOKBACK_DAYS 条记录。"""
        with sqlite3.connect(self.main_engine.db_path) as conn:
            df = pd.read_sql(
                """SELECT date, open, high, low, close, volume
                   FROM stock_daily WHERE symbol=?
                   ORDER BY date DESC LIMIT ?""",
                conn,
                params=(symbol, MAIN_LOOKBACK_DAYS + 100),  # 多取一点用于指标预热
            )
        return df.sort_values("date").reset_index(drop=True)

    # ── 通过 baostock 补拉熊市数据 ───────────────────────────────────────

    @staticmethod
    def _fetch_bear_data(symbol: str, start: str, end: str) -> pd.DataFrame:
        """通过 baostock 拉取指定区间的后复权日 K 数据。"""
        import baostock as bs

        prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
        bs_code = f"{prefix}.{symbol}"

        bs.login()
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount",
                start_date=start,
                end_date=end,
                frequency="d",
                adjustflag="1",
            )
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
        finally:
            bs.logout()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "turnover"])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]
        return df.reset_index(drop=True)

    # ── 核心：扫描单只股票，提取优质形态 ────────────────────────────────

    def _extract_patterns(
        self,
        df: pd.DataFrame,
        symbol: str,
        market_type: str,
    ) -> list[dict]:
        """扫描 df，返回满足条件的形态记录列表。"""
        if len(df) < PATTERN_WINDOW + 10:
            return []

        # 确保数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        if len(df) < 115:  # MA114 最小需求
            return []

        # 预加载大盘基准（整个批次只加载一次）
        try:
            benchmark = _ensure_benchmark_loaded()
        except Exception:
            benchmark = {}

        # 计算信号
        try:
            signals = _any_signal(df)
        except Exception as exc:
            logger.debug(f"[{symbol}] 信号计算失败: {exc}")
            return []

        records = []
        for idx in range(PATTERN_WINDOW, len(df) - 5):  # 至少留5天未来
            if not signals.iloc[idx]:
                continue

            window_df = df.iloc[idx - PATTERN_WINDOW: idx].copy()
            if len(window_df) < PATTERN_WINDOW:
                continue

            pattern = _normalize_pattern(window_df)
            s5, s30, s90, score = _calc_future_score(df, idx, self.weights, benchmark=benchmark)

            if score < self.score_threshold:
                continue

            records.append({
                "symbol":       symbol,
                "signal_date":  df["date"].iloc[idx],
                "market_type":  market_type,
                "pattern_json": json.dumps(pattern),
                "score_5d":     s5,
                "score_30d":    s30,
                "score_90d":    s90,
                "future_score": score,
            })

        return records

    def _save_records(self, records: list[dict]) -> int:
        """批量写入形态库，已存在的跳过（UNIQUE 约束）。"""
        if not records:
            return 0
        saved = 0
        with sqlite3.connect(self.lib_db_path) as conn:
            for r in records:
                try:
                    conn.execute(
                        """INSERT OR IGNORE INTO patterns
                           (symbol, signal_date, market_type, pattern_json,
                            score_5d, score_30d, score_90d, future_score)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (r["symbol"], r["signal_date"], r["market_type"],
                         r["pattern_json"], r["score_5d"], r["score_30d"],
                         r["score_90d"], r["future_score"]),
                    )
                    saved += 1
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
        return saved

    # ── 公开接口：构建主数据形态库 ───────────────────────────────────────

    def build_from_main(self, symbols: list[str] | None = None) -> int:
        """从主库近300天数据构建形态库。

        Args:
            symbols: 指定股票列表；None 则处理全部。

        Returns:
            入库形态总数。
        """
        if symbols is None:
            symbols = self.main_engine.get_local_symbols()

        total = 0
        for i, symbol in enumerate(symbols):
            try:
                df = self._load_main_data(symbol)
                records = self._extract_patterns(df, symbol, "bull")
                saved = self._save_records(records)
                total += saved
            except Exception as exc:
                logger.debug(f"[{symbol}] 主数据处理失败: {exc}")

            if (i + 1) % 500 == 0:
                logger.info(f"主数据进度 {i+1}/{len(symbols)}，累计入库 {total} 条形态")

        logger.info(f"主数据形态库构建完成，共入库 {total} 条")
        return total

    # ── 公开接口：补充熊市数据 ───────────────────────────────────────────

    def build_from_bear_periods(
        self,
        symbols: list[str] | None = None,
        periods: list[tuple[str, str, str]] | None = None,
    ) -> int:
        """通过 baostock 拉取熊市区间数据，补充形态库。

        Args:
            symbols: 股票列表；None 则使用主库全部股票。
            periods: [(start, end, market_type), ...]；None 则使用默认四段。

        Returns:
            入库形态总数。
        """
        if symbols is None:
            symbols = self.main_engine.get_local_symbols()
        if periods is None:
            periods = BEAR_PERIODS

        total = 0
        for start, end, mtype in periods:
            logger.info(f"开始处理熊市区间 {mtype}（{start} ~ {end}），共 {len(symbols)} 只股票")
            period_count = 0

            for i, symbol in enumerate(symbols):
                try:
                    df = self._fetch_bear_data(symbol, start, end)
                    if df.empty:
                        continue
                    records = self._extract_patterns(df, symbol, mtype)
                    saved = self._save_records(records)
                    period_count += saved
                except Exception as exc:
                    logger.debug(f"[{symbol}][{mtype}] 处理失败: {exc}")

                if (i + 1) % 200 == 0:
                    logger.info(f"  [{mtype}] 进度 {i+1}/{len(symbols)}，本段入库 {period_count} 条")

            logger.info(f"{mtype} 完成，入库 {period_count} 条形态")
            total += period_count

        logger.info(f"熊市形态库构建完成，共入库 {total} 条")
        return total

    # ── 公开接口：查询形态库 ────────────────────────────────────────────

    def query(
        self,
        market_types: list[str] | None = None,
        min_score: float = DEFAULT_SCORE_THRESHOLD,
        limit: int = 5000,
        random_sample: bool = True,
    ) -> pd.DataFrame:
        """查询形态库，返回 DataFrame。

        Args:
            market_types: 过滤市场类型；None 表示全部。
            min_score: 最低综合得分。
            limit: 最大返回条数。
            random_sample: 为 True 时随机采样（保证分布均匀，避免只取高分形态导致预测偏高）。
        """
        where_clauses = [f"future_score >= {min_score}"]
        if market_types:
            placeholders = ",".join(["?"] * len(market_types))
            where_clauses.append(f"market_type IN ({placeholders})")
        where = " AND ".join(where_clauses)
        params = market_types or []

        order = "ORDER BY RANDOM()" if random_sample else "ORDER BY future_score DESC"
        sql = f"""
            SELECT symbol, signal_date, market_type,
                   pattern_json, score_5d, score_30d, score_90d, future_score
            FROM patterns
            WHERE {where}
            {order}
            LIMIT {limit}
        """
        with sqlite3.connect(self.lib_db_path) as conn:
            df = pd.read_sql(sql, conn, params=params)
        return df

    def stats(self) -> dict:
        """返回形态库统计信息。"""
        with sqlite3.connect(self.lib_db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM patterns").fetchone()[0]
            by_type = conn.execute(
                "SELECT market_type, COUNT(*), AVG(future_score) FROM patterns GROUP BY market_type"
            ).fetchall()
            score_dist = conn.execute(
                """SELECT
                     SUM(CASE WHEN future_score >= 0.8 THEN 1 ELSE 0 END) AS high,
                     SUM(CASE WHEN future_score >= 0.6 AND future_score < 0.8 THEN 1 ELSE 0 END) AS mid,
                     SUM(CASE WHEN future_score < 0.6 THEN 1 ELSE 0 END) AS low
                   FROM patterns"""
            ).fetchone()
        return {
            "total": total,
            "by_market_type": {r[0]: {"count": r[1], "avg_score": round(r[2], 3)} for r in by_type},
            "score_distribution": {"high(>=0.8)": score_dist[0], "mid(0.6~0.8)": score_dist[1], "low(<0.6)": score_dist[2]},
        }
