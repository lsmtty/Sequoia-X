"""形态匹配模块。

两步匹配：
  Step1：DTW 距离快速粗筛（只用 close 归一化序列，速度快）
  Step2：K 线图像渲染 + SSIM 结构相似度精排

输出：
  MatchResult 列表，每条包含：
    - 当前股票代码 / 信号日
    - 匹配到的历史形态（来自形态库）
    - dtw_score、image_score、final_score
    - 历史形态的 future_score（反映当时之后的表现）
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

# ── 配置 ─────────────────────────────────────────────────────────────────

DTW_TOPK        = 20    # DTW 粗筛保留 Top-K 候选
FINAL_TOPK      = 5     # 最终每只股票保留 Top-K 匹配结果
DTW_WEIGHT      = 0.4   # DTW 得分权重
IMAGE_WEIGHT    = 0.6   # 图像得分权重


# ── 数据结构 ──────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    """单条匹配结果。"""
    query_symbol:   str
    query_date:     str         # 当前信号日期
    lib_symbol:     str         # 匹配到的历史股票
    lib_signal_date: str        # 历史信号日期
    lib_market_type: str        # 市场类型
    lib_future_score: float     # 历史形态的未来表现得分
    lib_score_5d:   float
    lib_score_30d:  float
    lib_score_90d:  float
    dtw_score:      float       # DTW 相似度得分 0~1（越高越好）
    image_score:    float       # 图像相似度得分 0~1
    final_score:    float       # 综合得分


# ── Step1：DTW 粗筛 ───────────────────────────────────────────────────────

def _extract_close_seq(pattern_json: str) -> np.ndarray:
    """从形态JSON提取 close 序列（已归一化）。"""
    arr = json.loads(pattern_json)
    return np.array([row[3] for row in arr], dtype=np.float32)  # index 3 = close


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """简单 DTW 距离计算（无 sakura 窗口，适合等长序列）。"""
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(float(a[i - 1]) - float(b[j - 1]))
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def _dtw_distance_fast(a: np.ndarray, b: np.ndarray) -> float:
    """优先使用 fastdtw，失败则回退到简单 DTW。"""
    try:
        from fastdtw import fastdtw  # type: ignore
        dist, _ = fastdtw(a, b, dist=lambda x, y: abs(x - y))
        return float(dist)
    except ImportError:
        return _dtw_distance(a, b)


def dtw_topk(
    query_seq: np.ndarray,
    library_df: pd.DataFrame,
    topk: int = DTW_TOPK,
) -> pd.DataFrame:
    """对形态库做 DTW 粗筛，返回距离最小的 TopK 候选行。

    Args:
        query_seq: 当前股票归一化 close 序列（长度 PATTERN_WINDOW）
        library_df: PatternLibrary.query() 返回的 DataFrame
        topk: 保留候选数量

    Returns:
        带 dtw_dist 列的 TopK 行 DataFrame
    """
    if library_df.empty:
        return library_df

    dists = []
    for row in library_df.itertuples():
        lib_seq = _extract_close_seq(row.pattern_json)
        if len(lib_seq) != len(query_seq):
            # 长度不一致时截断/补齐到相同长度
            min_len = min(len(lib_seq), len(query_seq))
            lib_seq = lib_seq[:min_len]
            q = query_seq[:min_len]
        else:
            q = query_seq
        dists.append(_dtw_distance_fast(q, lib_seq))

    library_df = library_df.copy()
    library_df["dtw_dist"] = dists

    # 距离转得分：1 - normalize(dist)
    max_dist = max(dists) if dists else 1.0
    library_df["dtw_score"] = 1.0 - (library_df["dtw_dist"] / max_dist)

    return library_df.nsmallest(topk, "dtw_dist").reset_index(drop=True)


# ── Step2：图像渲染 + SSIM 精排 ───────────────────────────────────────────

def _render_pattern_to_array(
    pattern_data: list[list[float]],
    img_size: tuple[int, int] = (128, 64),
) -> np.ndarray:
    """将形态数据渲染为灰度图像 numpy 数组。

    pattern_data: [[open,high,low,close,vol], ...]
    返回 shape=(H, W) 的 float32 数组，值域 [0,1]
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logger.warning("matplotlib 未安装，图像得分将返回 0")
        return np.zeros(img_size, dtype=np.float32)

    w, h = img_size
    fig, ax = plt.subplots(figsize=(w / 32, h / 32), dpi=32)
    ax.set_facecolor("black")
    fig.patch.set_facecolor("black")
    ax.axis("off")

    closes = [row[3] for row in pattern_data]
    highs  = [row[1] for row in pattern_data]
    lows   = [row[2] for row in pattern_data]
    opens  = [row[0] for row in pattern_data]

    y_min = min(lows)
    y_max = max(highs)
    y_range = y_max - y_min if y_max != y_min else 1.0

    n = len(pattern_data)
    bar_w = 0.6

    for i in range(n):
        o, c = opens[i], closes[i]
        color = "#ef232a" if c >= o else "#14b143"
        # 影线
        ax.plot([i, i], [lows[i], highs[i]], color=color, linewidth=0.5)
        # 实体
        ax.add_patch(
            mpatches.Rectangle(
                (i - bar_w / 2, min(o, c)),
                bar_w,
                abs(c - o) + 1e-9,
                color=color,
                linewidth=0,
            )
        )

    ax.set_xlim(-1, n)
    ax.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.05)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                facecolor="black", dpi=32)
    plt.close(fig)
    buf.seek(0)

    try:
        from PIL import Image
        img = Image.open(buf).convert("L").resize(img_size)
        arr = np.array(img, dtype=np.float32) / 255.0
    except ImportError:
        logger.warning("Pillow 未安装，图像得分将返回 0")
        arr = np.zeros(img_size, dtype=np.float32)

    return arr


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """简化版 SSIM（结构相似度），返回 [0, 1]。"""
    try:
        from skimage.metrics import structural_similarity  # type: ignore
        score, _ = structural_similarity(a, b, full=True, data_range=1.0)
        return float(max(0.0, score))
    except ImportError:
        pass

    # 手动实现简化 SSIM
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = a.std()
    sigma_b = b.std()
    sigma_ab = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / \
           ((mu_a ** 2 + mu_b ** 2 + c1) * (sigma_a ** 2 + sigma_b ** 2 + c2))
    return float(max(0.0, min(1.0, ssim)))


def image_rerank(
    query_pattern: list[list[float]],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """对 DTW 候选做图像 SSIM 精排。

    Args:
        query_pattern: 当前形态数据 [[o,h,l,c,v], ...]
        candidates: dtw_topk 返回的候选 DataFrame

    Returns:
        添加 image_score 和 final_score 列的 DataFrame
    """
    if candidates.empty:
        return candidates

    query_img = _render_pattern_to_array(query_pattern)
    image_scores = []

    for row in candidates.itertuples():
        lib_pattern = json.loads(row.pattern_json)
        lib_img = _render_pattern_to_array(lib_pattern)
        score = _ssim(query_img, lib_img)
        image_scores.append(score)

    candidates = candidates.copy()
    candidates["image_score"] = image_scores
    candidates["final_score"] = (
        DTW_WEIGHT    * candidates["dtw_score"]
        + IMAGE_WEIGHT * candidates["image_score"]
    )
    return candidates.sort_values("final_score", ascending=False).reset_index(drop=True)


# ── 对外主接口 ────────────────────────────────────────────────────────────

def match_stock(
    symbol: str,
    query_pattern: list[list[float]],
    query_date: str,
    library_df: pd.DataFrame,
    topk: int = FINAL_TOPK,
) -> list[MatchResult]:
    """对单只股票做完整两步形态匹配。

    Args:
        symbol: 当前股票代码
        query_pattern: 当前形态数据（30×5）
        query_date: 当前信号日期（通常为今日）
        library_df: 形态库 DataFrame（PatternLibrary.query() 返回）
        topk: 最终返回匹配数量

    Returns:
        MatchResult 列表，按 final_score 降序
    """
    if library_df.empty:
        return []

    # 提取 query close 序列
    query_seq = np.array([row[3] for row in query_pattern], dtype=np.float32)

    # Step1: DTW 粗筛
    try:
        candidates = dtw_topk(query_seq, library_df, topk=DTW_TOPK)
    except Exception as exc:
        logger.warning(f"[{symbol}] DTW 粗筛失败: {exc}")
        return []

    # Step2: 图像精排
    try:
        ranked = image_rerank(query_pattern, candidates)
    except Exception as exc:
        logger.warning(f"[{symbol}] 图像精排失败: {exc}")
        ranked = candidates.copy()
        ranked["image_score"] = 0.0
        ranked["final_score"] = ranked.get("dtw_score", 0.0)

    results = []
    for row in ranked.head(topk).itertuples():
        results.append(MatchResult(
            query_symbol    = symbol,
            query_date      = query_date,
            lib_symbol      = row.symbol,
            lib_signal_date = row.signal_date,
            lib_market_type = row.market_type,
            lib_future_score = row.future_score,
            lib_score_5d    = row.score_5d,
            lib_score_30d   = row.score_30d,
            lib_score_90d   = row.score_90d,
            dtw_score       = round(float(row.dtw_score), 4),
            image_score     = round(float(row.image_score), 4),
            final_score     = round(float(row.final_score), 4),
        ))

    return results


def match_batch(
    candidates: list[tuple[str, list[list[float]]]],
    query_date: str,
    library_df: pd.DataFrame,
    topk: int = FINAL_TOPK,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, list[MatchResult]]:
    """批量对候选股票做形态匹配。

    Args:
        candidates: [(symbol, pattern_data), ...]
        query_date: 信号日期
        library_df: 形态库
        topk: 每只保留匹配数
        on_progress: 进度回调 (current, total)

    Returns:
        {symbol: [MatchResult, ...]}
    """
    results: dict[str, list[MatchResult]] = {}
    total = len(candidates)

    for i, (symbol, pattern) in enumerate(candidates):
        results[symbol] = match_stock(symbol, pattern, query_date, library_df, topk)
        if on_progress:
            on_progress(i + 1, total)

    return results


def summarize_match(matches: list[MatchResult]) -> dict:
    """汇总单只股票的匹配结果，返回可读摘要。"""
    if not matches:
        return {"final_score": 0.0, "expected_5d": 0.0, "expected_30d": 0.0, "expected_90d": 0.0, "top_matches": []}

    # 加权平均历史未来表现（按 final_score 加权）
    weights = np.array([m.final_score for m in matches])
    w_sum = weights.sum()
    if w_sum == 0:
        weights = np.ones(len(matches)) / len(matches)
    else:
        weights = weights / w_sum

    avg_score    = sum(m.final_score     * w for m, w in zip(matches, weights))
    avg_5d       = sum(m.lib_score_5d    * w for m, w in zip(matches, weights))
    avg_30d      = sum(m.lib_score_30d   * w for m, w in zip(matches, weights))
    avg_90d      = sum(m.lib_score_90d   * w for m, w in zip(matches, weights))

    top = [{
        "lib_symbol":   m.lib_symbol,
        "lib_date":     m.lib_signal_date,
        "market_type":  m.lib_market_type,
        "dtw":          m.dtw_score,
        "image":        m.image_score,
        "final":        m.final_score,
        "future_score": m.lib_future_score,
    } for m in matches[:3]]

    return {
        "final_score":    round(avg_score, 4),
        "expected_5d":    round(avg_5d,    4),
        "expected_30d":   round(avg_30d,   4),
        "expected_90d":   round(avg_90d,   4),
        "top_matches":    top,
    }
