"""主测试脚本：选股 → 形态匹配 → HTML 报告 → 企业微信推送。

用法：
    # 第一次运行（或形态库为空时需先构建）：
    .venv/bin/python run_zhixing_test.py --build-library

    # 日常运行（形态库已有数据）：
    .venv/bin/python run_zhixing_test.py

    # 跳过形态匹配，只出选股报告：
    .venv/bin/python run_zhixing_test.py --no-match

    # 运行并推送 Top10 到企业微信（需配置 WECOM_WEBHOOK_KEY 环境变量）：
    .venv/bin/python run_zhixing_test.py --notify

环境变量：
    WECOM_WEBHOOK_KEY  企业微信群机器人的 key（Webhook URL 中的 key 参数）
"""

import argparse
import os
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

os.environ.setdefault("FEISHU_WEBHOOK_URL", "https://placeholder.local")

from sequoia_x.core.config import Settings
from sequoia_x.core.logger import get_logger
from sequoia_x.data.engine import DataEngine
from sequoia_x.pattern_library import PatternLibrary
from sequoia_x.pattern_matcher import match_batch, summarize_match
from sequoia_x.report import generate_report, generate_top10_report
from sequoia_x.strategy.zhixing_kdj_trend import ZhixingKdjTrendStrategy
from sequoia_x.strategy.zhixing_brick_reversal import ZhixingBrickReversalStrategy

logger = get_logger(__name__)
PATTERN_WINDOW = 30

# 砖型图策略名称（用于标记 brick_signal 字段）
BRICK_STRATEGY_NAME = "知行砖型图红绿柱翻转"


# ── 辅助：截取归一化 K 线形态 ────────────────────────────────────────────

def get_query_pattern(engine: DataEngine, symbol: str) -> list[list[float]] | None:
    """截取当前股票最近 PATTERN_WINDOW 根K线并归一化，用于匹配。"""
    with sqlite3.connect(engine.db_path) as conn:
        df = pd.read_sql(
            "SELECT open,high,low,close,volume FROM stock_daily "
            "WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn,
            params=(symbol, PATTERN_WINDOW),
        )
    if len(df) < PATTERN_WINDOW:
        return None

    df = df.iloc[::-1].reset_index(drop=True)
    base = float(df["close"].iloc[0])
    if base == 0:
        return None

    v_min   = float(df["volume"].min())
    v_max   = float(df["volume"].max())
    v_range = v_max - v_min if v_max != v_min else 1.0

    result = []
    for row in df.itertuples(index=False):
        result.append([
            round(row.open   / base, 6),
            round(row.high   / base, 6),
            round(row.low    / base, 6),
            round(row.close  / base, 6),
            round((row.volume - v_min) / v_range, 6),
        ])
    return result


# ── 辅助：通过 baostock 批量查股票名称和所属板块 ─────────────────────────

def _fetch_stock_info(symbols: list[str]) -> dict[str, dict]:
    """返回 {symbol: {"name": ..., "sector": ...}}。

    板块（industry）来自 baostock query_stock_basic 的 industry 字段。
    若 baostock 不可用，降级返回空名称。
    """
    info: dict[str, dict] = {s: {"name": s, "sector": "未知板块"} for s in symbols}
    try:
        import baostock as bs
        bs.login()
        for symbol in symbols:
            prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
            rs = bs.query_stock_basic(code=f"{prefix}.{symbol}")
            while rs.next():
                row = rs.get_row_data()
                # fields: code, code_name, ipoDate, outDate, type, status
                # query_stock_basic 返回字段顺序：code,code_name,ipoDate,outDate,type,status
                name = row[1] if len(row) > 1 else symbol
                info[symbol] = {"name": name, "sector": "A股"}
        bs.logout()
    except Exception as exc:
        logger.warning(f"股票信息查询失败（降级使用代码）: {exc}")
    return info


def _fetch_stock_industry(symbols: list[str]) -> dict[str, str]:
    """通过 baostock query_stock_industry 获取申万行业分类。

    返回 {symbol: industry_name}，失败时返回空字符串。
    """
    result: dict[str, str] = {}
    try:
        import baostock as bs
        from datetime import date as _date
        today = _date.today().strftime("%Y-%m-%d")
        bs.login()
        for symbol in symbols:
            prefix = "sh" if symbol.startswith(("6", "9")) else "sz"
            rs = bs.query_stock_industry(code=f"{prefix}.{symbol}", date=today)
            while rs.next():
                row = rs.get_row_data()
                # fields: updateDate, code, code_name, industry, industryClassification
                if len(row) >= 4:
                    result[symbol] = row[3]  # industry
        bs.logout()
    except Exception as exc:
        logger.warning(f"行业信息查询失败: {exc}")
    return result


# ── 辅助：构建 Top10 股票信息列表 ────────────────────────────────────────

def build_top10(
    match_scores: dict[str, dict],
    results: dict[str, list[str]],
    topk: int = 10,
) -> list[dict]:
    """按形态匹配分降序，取 Top-K 构建带元信息的列表。

    Returns:
        list of dict，每条包含：
        symbol, name, sector, strategy, brick_signal,
        match_score, expected_5d, expected_30d, expected_90d
    """
    # 股票 → 策略 映射（一只股票可能被多个策略选中，取第一个）
    sym_to_strategy: dict[str, str] = {}
    brick_symbols: set[str] = set(results.get(BRICK_STRATEGY_NAME, []))
    for strategy_name, syms in results.items():
        for s in syms:
            if s not in sym_to_strategy:
                sym_to_strategy[s] = strategy_name

    # 按 match_score 降序排列
    sorted_syms = sorted(
        match_scores.items(),
        key=lambda x: x[1].get("final_score", 0.0),
        reverse=True,
    )[:topk]

    if not sorted_syms:
        return []

    symbols = [s for s, _ in sorted_syms]

    # 批量查名称和板块
    print("📋 查询股票名称和板块信息...")
    stock_info   = _fetch_stock_info(symbols)
    industry_map = _fetch_stock_industry(symbols)

    top10 = []
    for symbol, sc in sorted_syms:
        info    = stock_info.get(symbol, {})
        sector  = industry_map.get(symbol) or info.get("sector", "未知板块")
        top10.append({
            "symbol":       symbol,
            "name":         info.get("name", symbol),
            "sector":       sector,
            "strategy":     sym_to_strategy.get(symbol, ""),
            "brick_signal": symbol in brick_symbols,
            "match_score":  sc.get("final_score", 0.0),
            "expected_5d":  sc.get("expected_5d",  0.0),
            "expected_30d": sc.get("expected_30d", 0.0),
            "expected_90d": sc.get("expected_90d", 0.0),
        })
    return top10


# ── 主流程 ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sequoia-X 选股 + 形态匹配 + HTML 报告")
    parser.add_argument("--build-library", action="store_true",
                        help="构建/更新主数据形态库（从近300天数据扫描）")
    parser.add_argument("--build-bear", action="store_true",
                        help="补充熊市数据到形态库（需联网拉取历史，耗时较长）")
    parser.add_argument("--no-match", action="store_true",
                        help="跳过形态匹配，只生成选股报告")
    parser.add_argument("--notify", action="store_true",
                        help="生成 Top10 精简报告并推送到企业微信")
    parser.add_argument("--no-open", action="store_true",
                        help="不自动打开浏览器（定时任务场景使用）")
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

    stats = lib.stats()
    print(f"📊 形态库统计：共 {stats['total']} 条形态")
    for mtype, info in stats["by_market_type"].items():
        print(f"   {mtype}: {info['count']} 条，平均得分 {info['avg_score']}")
    print()

    # ── Step3（可选）：形态匹配 ─────────────────────────────────────────
    match_scores: dict[str, dict] = {}

    if not args.no_match and stats["total"] > 0:
        print("🔍 加载形态库...")
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

        total_c = len(candidates_with_patterns)
        print(f"   共 {total_c} 只候选股票需要匹配\n")

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

    # ── Step4：生成完整 HTML 报告 ─────────────────────────────────────────
    output_path = Path(__file__).parent / "report.html"
    print("📄 生成完整 HTML 报告...")
    report_file = generate_report(
        results=results,
        db_path=settings.db_path,
        output_path=str(output_path),
        match_scores=match_scores,
    )
    print(f"✅ 报告已生成：{report_file}\n")

    if not args.no_open:
        subprocess.run(["open", report_file], check=False)

    # ── Step5（可选）：生成 Top10 报告 + 企业微信推送 ─────────────────────
    if args.notify and match_scores:
        print("📲 构建 Top10 列表...")
        top10 = build_top10(match_scores, results, topk=10)

        if not top10:
            print("⚠️  无匹配结果，跳过推送\n")
            return

        # 生成 Top10 精简 HTML
        top10_path = Path(__file__).parent / "report_top10.html"
        print("📄 生成 Top10 精简报告...")
        top10_report = generate_top10_report(
            top10=top10,
            db_path=settings.db_path,
            output_path=str(top10_path),
        )
        print(f"✅ Top10 报告已生成：{top10_report}\n")

        # 企业微信推送
        from sequoia_x.notify.wecom import WecomNotifier
        notifier = WecomNotifier()
        print("📲 推送到企业微信...")
        notifier.send_top10(top10=top10, report_html_path=top10_report)
        print("✅ 企业微信推送完成\n")

    elif args.notify and not match_scores:
        print("⚠️  形态匹配未运行（--no-match 或形态库为空），无法推送 Top10\n")


if __name__ == "__main__":
    main()
