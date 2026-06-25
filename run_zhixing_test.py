"""主测试脚本：选股 → 形态匹配 → HTML 报告。

用法：
    # 第一次运行（或形态库为空时需先构建）：
    .venv/bin/python run_zhixing_test.py --build-library

    # 日常运行（形态库已有数据）：
    .venv/bin/python run_zhixing_test.py

    # 跳过形态匹配，只出选股报告：
    .venv/bin/python run_zhixing_test.py --no-match
"""

import argparse
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("FEISHU_WEBHOOK_URL", "https://placeholder.local")

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.pattern_library import PatternLibrary
from sequoia_x.pattern_matcher import match_batch, summarize_match
from sequoia_x.report import generate_report
from sequoia_x.strategy.zhixing_kdj_trend import ZhixingKdjTrendStrategy
from sequoia_x.strategy.zhixing_brick_reversal import ZhixingBrickReversalStrategy

logger = get_logger(__name__)
PATTERN_WINDOW = 30


def get_query_pattern(engine: DataEngine, symbol: str) -> list[list[float]] | None:
    """截取当前股票最近 PATTERN_WINDOW 根K线并归一化，用于匹配。"""
    import sqlite3
    import pandas as pd

    with sqlite3.connect(engine.db_path) as conn:
        df = pd.read_sql(
            "SELECT open,high,low,close,volume FROM stock_daily "
            "WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn,
            params=(symbol, PATTERN_WINDOW),
        )
    if len(df) < PATTERN_WINDOW:
        return None

    # SQL 按 date DESC 返回，翻转为正序（最早在前）
    df = df.iloc[::-1].reset_index(drop=True)

    base = float(df["close"].iloc[0])
    if base == 0:
        return None

    v_min  = float(df["volume"].min())
    v_max  = float(df["volume"].max())
    v_range = v_max - v_min if v_max != v_min else 1.0

    result = []
    for i, row in enumerate(df.itertuples(index=False)):
        result.append([
            round(row.open  / base, 6),
            round(row.high  / base, 6),
            round(row.low   / base, 6),
            round(row.close / base, 6),
            round((row.volume - v_min) / v_range, 6),
        ])
    return result


def main():
    parser = argparse.ArgumentParser(description="Sequoia-X 选股 + 形态匹配 + HTML 报告")
    parser.add_argument("--build-library", action="store_true",
                        help="构建/更新主数据形态库（从近300天数据扫描）")
    parser.add_argument("--build-bear", action="store_true",
                        help="补充熊市数据到形态库（需联网拉取历史，耗时较长）")
    parser.add_argument("--no-match", action="store_true",
                        help="跳过形态匹配，只生成选股报告")
    args = parser.parse_args()

    settings = Settings()
    engine   = DataEngine(settings)

    symbols = engine.get_local_symbols()
    if not symbols:
        print("\n❌ 本地数据库为空，请先执行：")
        print("   .venv/bin/python main.py --backfill")
        sys.exit(1)

    print(f"\n✅ 本地数据库已有 {len(symbols)} 只股票\n")

    # ── Step1：运行选股策略 ─────────────────────────────────────────────
    strategies = [
        ("知行KDJ低位+趋势",    ZhixingKdjTrendStrategy(engine=engine, settings=settings)),
        ("知行砖型图红绿柱翻转", ZhixingBrickReversalStrategy(engine=engine, settings=settings)),
    ]

    results: dict[str, list[str]] = {}
    all_candidates: list[str] = []
    for name, strategy in strategies:
        print(f"▶ 运行策略：{name}")
        selected = strategy.run()
        results[name] = selected
        all_candidates.extend(selected)
        print(f"  → 选出 {len(selected)} 只\n")

    # ── Step2（可选）：形态库构建 ────────────────────────────────────────
    lib = PatternLibrary(settings)

    if args.build_library:
        print("📚 构建主数据形态库（近300天）...")
        n = lib.build_from_main()
        print(f"  → 入库 {n} 条形态\n")

    if args.build_bear:
        print("📚 补充熊市形态数据（联网拉取，约需数分钟）...")
        n = lib.build_from_bear_periods()
        print(f"  → 入库 {n} 条熊市形态\n")

    # 打印形态库统计
    stats = lib.stats()
    print(f"📊 形态库统计：共 {stats['total']} 条形态")
    for mtype, info in stats["by_market_type"].items():
        print(f"   {mtype}: {info['count']} 条，平均得分 {info['avg_score']}")
    print()

    # ── Step3（可选）：形态匹配 ─────────────────────────────────────────
    match_scores: dict[str, dict] = {}

    if not args.no_match and stats["total"] > 0:
        print("🔍 加载形态库...")
        # min_score=0 + random_sample=True：随机采样全量形态（含低分），保证分布均匀
        library_df = lib.query(min_score=0.0, limit=5000, random_sample=True)
        print(f"   加载 {len(library_df)} 条形态（随机均匀采样）\n")

        print("🔍 开始形态匹配...")
        today_str = date.today().strftime("%Y-%m-%d")

        unique_candidates = list(dict.fromkeys(all_candidates))
        candidates_with_patterns: list[tuple[str, list[list[float]]]] = []
        for sym in unique_candidates:
            pattern = get_query_pattern(engine, sym)
            if pattern:
                candidates_with_patterns.append((sym, pattern))

        total = len(candidates_with_patterns)
        print(f"   共 {total} 只候选股票需要匹配\n")

        def on_progress(cur: int, tot: int) -> None:
            if cur % 20 == 0 or cur == tot:
                print(f"   进度 {cur}/{tot}...", end="\r")

        raw_matches = match_batch(
            candidates_with_patterns,
            query_date=today_str,
            library_df=library_df,
            on_progress=on_progress,
        )
        print()

        for sym, matches in raw_matches.items():
            match_scores[sym] = summarize_match(matches)

        sorted_scores = sorted(
            match_scores.items(),
            key=lambda x: x[1]["final_score"],
            reverse=True,
        )
        print("\n🏆 形态匹配 Top10：")
        for sym, sc in sorted_scores[:10]:
            print(f"   {sym}  综合:{sc['final_score']:.3f}  "
                  f"5日:{sc['expected_5d']:.2f}  "
                  f"30日:{sc['expected_30d']:.2f}  "
                  f"90日:{sc['expected_90d']:.2f}")
        print()

    elif not args.no_match and stats["total"] == 0:
        print("⚠️  形态库为空，跳过匹配。请先运行：")
        print("   .venv/bin/python run_zhixing_test.py --build-library\n")

    # ── Step4：生成 HTML 报告 ────────────────────────────────────────────
    output_path = Path(__file__).parent / "report.html"
    print("📄 生成 HTML 报告...")
    report_file = generate_report(
        results=results,
        db_path=settings.db_path,
        output_path=str(output_path),
        match_scores=match_scores,
    )
    print(f"✅ 报告已生成：{report_file}\n")

    subprocess.run(["open", report_file], check=False)


if __name__ == "__main__":
    main()
