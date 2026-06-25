"""HTML 选股报告生成模块。

为每只选股结果生成包含以下内容的 HTML 报告：
- 汇总表：策略名、股票代码、最新收盘价、今日涨幅、形态匹配得分（可选）
- 每只股票的交互式 K 线图 + 知行趋势线/多空线 + KDJ(J值) + 砖型图
- 形态匹配区：显示 DTW+图像综合得分、预期5/30/90日涨幅、Top3历史相似形态
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# ── 指标计算（与策略内保持一致）──────────────────────────────────────────


def _sma_td(series: pd.Series, n: int, m: int) -> pd.Series:
    """通达信 SMA 递推算法。"""
    result = np.zeros(len(series))
    values = series.values
    result[0] = values[0]
    for i in range(1, len(values)):
        result[i] = (m * values[i] + (n - m) * result[i - 1]) / n
    return pd.Series(result, index=series.index)


def _calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """计算知行趋势线、多空线、KDJ-J 值、砖型图。"""
    df = df.copy()

    ema10 = df["close"].ewm(span=10, adjust=False).mean()
    df["trend_line"] = ema10.ewm(span=10, adjust=False).mean()

    df["bull_bear"] = (
        df["close"].rolling(14).mean()
        + df["close"].rolling(28).mean()
        + df["close"].rolling(57).mean()
        + df["close"].rolling(114).mean()
    ) / 4

    n = 9
    llv = df["low"].rolling(n).min()
    hhv = df["high"].rolling(n).max()
    denom = (hhv - llv).replace(0, np.nan)
    rsv = ((df["close"] - llv) / denom * 100).fillna(50)
    k = _sma_td(rsv, 3, 1)
    d = _sma_td(k, 3, 1)
    df["kdj_j"] = 3 * k - 2 * d

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
    df["brick"] = np.where(var6a > 4, var6a - 4, 0)

    return df


# ── 数据加载 ──────────────────────────────────────────────────────────────


def _load_stock_data(db_path: str, symbol: str, n_bars: int = 120) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT * FROM stock_daily WHERE symbol=? ORDER BY date DESC LIMIT ?",
            conn,
            params=(symbol, n_bars),
        )
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _stock_chart_json(df: pd.DataFrame, symbol: str) -> dict:
    """生成单只股票的 ECharts option 数据（JSON 可序列化）。"""
    df = _calc_indicators(df)
    tail = df.tail(60).reset_index(drop=True)

    dates = tail["date"].tolist()
    kline = [[
        round(row.open, 2), round(row.close, 2),
        round(row.low, 2),  round(row.high, 2)
    ] for row in tail.itertuples()]

    trend  = [round(v, 2) if not np.isnan(v) else None for v in tail["trend_line"]]
    bull   = [round(v, 2) if not np.isnan(v) else None for v in tail["bull_bear"]]
    kdj_j  = [round(v, 2) if not np.isnan(v) else None for v in tail["kdj_j"]]
    brick  = [round(v, 4) if not np.isnan(v) else None for v in tail["brick"]]
    brick_prev = [None] + brick[:-1]
    brick_colors = [
        "#ef232a" if (b is not None and p is not None and b > p) else "#14b143"
        for b, p in zip(brick, brick_prev)
    ]

    last = tail.iloc[-1]
    prev = tail.iloc[-2]
    pct = (last["close"] / prev["close"] - 1) * 100 if prev["close"] else 0

    return {
        "symbol":      symbol,
        "last_close":  round(float(last["close"]), 2),
        "pct":         round(float(pct), 2),
        "dates":       dates,
        "kline":       kline,
        "trend":       trend,
        "bull":        bull,
        "kdj_j":       kdj_j,
        "brick":       brick,
        "brick_colors": brick_colors,
    }


# ── 汇总 chip 渲染 ────────────────────────────────────────────────────────


def _render_summary_group(strategy_name: str, rows: list[dict]) -> str:
    chips = ""
    for r in rows:
        pct = r["pct"]
        cls = "up" if pct > 0 else "down" if pct < 0 else "flat"
        sign = "+" if pct > 0 else ""
        ms = r.get("match_score")
        ms_html = (
            f'<span class="ms-badge" title="形态匹配得分">'
            f'⚡{ms:.2f}</span>'
            if ms is not None else ""
        )
        chips += (
            f'<span class="chip" data-sym="{r["symbol"]}">'
            f'<span class="code">{r["symbol"]}</span>'
            f'<span class="pct {cls}">{sign}{pct}%</span>'
            f'{ms_html}'
            f'</span>'
        )
    return (
        f'<div class="strategy-group">'
        f'<div class="strategy-label">{strategy_name}（{len(rows)} 只）</div>'
        f'<div class="chips">{chips}</div>'
        f'</div>'
    )


# ── 报告生成主函数 ────────────────────────────────────────────────────────


def generate_report(
    results: dict[str, list[str]],
    db_path: str,
    output_path: str = "report.html",
    match_scores: dict[str, dict] | None = None,
) -> str:
    """生成 HTML 选股报告。

    Args:
        results: {策略名: [股票代码列表]}
        db_path: SQLite 数据库路径
        output_path: 输出 HTML 文件路径
        match_scores: {symbol: summarize_match() 返回的摘要字典}，可选。
                      若传入则在报告中显示形态匹配得分和预期涨幅。

    Returns:
        输出文件的绝对路径。
    """
    today_str = date.today().strftime("%Y-%m-%d")
    match_scores = match_scores or {}

    all_symbols: list[tuple[str, str]] = []
    for strategy_name, symbols in results.items():
        for s in symbols:
            all_symbols.append((strategy_name, s))

    charts_data: list[dict] = []
    summary_rows: list[dict] = []

    for strategy_name, symbol in all_symbols:
        try:
            df = _load_stock_data(db_path, symbol)
            if len(df) < 10:
                continue
            chart = _stock_chart_json(df, symbol)
            chart["strategy"] = strategy_name
            ms = match_scores.get(symbol, {})
            chart["match_score"]  = ms.get("final_score", None)
            chart["expected_5d"]  = ms.get("expected_5d", None)
            chart["expected_30d"] = ms.get("expected_30d", None)
            chart["expected_90d"] = ms.get("expected_90d", None)
            chart["top_matches"]  = ms.get("top_matches", [])
            charts_data.append(chart)
            summary_rows.append({
                "strategy":    strategy_name,
                "symbol":      symbol,
                "close":       chart["last_close"],
                "pct":         chart["pct"],
                "match_score": chart["match_score"],
            })
        except Exception:
            continue

    has_match = any(r.get("match_score") is not None for r in summary_rows)
    charts_json = json.dumps(charts_data, ensure_ascii=False)
    summary_html = "".join(
        _render_summary_group(
            sn, [r for r in summary_rows if r["strategy"] == sn]
        )
        for sn in results.keys()
    )

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>Sequoia-X 选股报告 | {today_str}</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
        background:#0d1117;color:#e6edf3}}
  header{{padding:24px 32px;border-bottom:1px solid #21262d}}
  header h1{{font-size:22px;font-weight:600}}
  header p{{font-size:13px;color:#8b949e;margin-top:4px}}
  .container{{max-width:1440px;margin:0 auto;padding:24px 32px}}
  .section-title{{font-size:15px;font-weight:600;color:#58a6ff;
                  margin-bottom:12px;padding-left:4px}}
  /* 汇总 chips */
  .summary-section{{margin-bottom:40px}}
  .strategy-group{{margin-bottom:24px}}
  .strategy-label{{font-size:13px;color:#8b949e;margin-bottom:8px;
                   padding:4px 10px;background:#161b22;border-radius:4px;
                   display:inline-block}}
  .chips{{display:flex;flex-wrap:wrap;gap:8px}}
  .chip{{display:inline-flex;align-items:center;gap:5px;padding:5px 11px;
         border-radius:20px;font-size:13px;background:#161b22;
         border:1px solid #30363d;cursor:pointer;transition:border-color .15s}}
  .chip:hover{{border-color:#58a6ff}}
  .chip .code{{font-weight:600}}
  .chip .pct{{font-size:12px}}
  .ms-badge{{font-size:11px;color:#f0c05a;background:rgba(240,192,90,.12);
             padding:1px 5px;border-radius:8px}}
  .up{{color:#ef232a}} .down{{color:#14b143}} .flat{{color:#8b949e}}
  /* 图表卡片 */
  .charts-section .section-title{{margin-bottom:20px}}
  .strategy-block{{margin-bottom:48px}}
  .strategy-block-title{{font-size:14px;font-weight:600;color:#f0f6fc;
    background:#161b22;border:1px solid #30363d;padding:8px 16px;
    border-radius:6px;margin-bottom:16px;display:inline-block}}
  .charts-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(680px,1fr));gap:16px}}
  .chart-card{{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px}}
  .chart-header{{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex-wrap:wrap}}
  .chart-header .sym{{font-size:15px;font-weight:700}}
  .chart-header .price{{font-size:13px;color:#8b949e}}
  .badge{{font-size:12px;padding:2px 8px;border-radius:10px;font-weight:600}}
  .badge.up{{background:rgba(239,35,42,.15);color:#ef232a}}
  .badge.down{{background:rgba(20,177,67,.15);color:#14b143}}
  /* 形态匹配面板 */
  .match-panel{{margin-top:8px;padding:8px 10px;background:#0d1117;
                border-radius:6px;border:1px solid #30363d;font-size:12px}}
  .match-panel .mp-title{{color:#f0c05a;font-weight:600;margin-bottom:6px}}
  .mp-scores{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:6px}}
  .mp-score-item{{display:flex;flex-direction:column;align-items:center}}
  .mp-score-item .label{{color:#8b949e;font-size:11px}}
  .mp-score-item .value{{font-size:14px;font-weight:700;color:#f0c05a}}
  .mp-exp{{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}}
  .mp-exp-item{{font-size:11px;color:#8b949e}}
  .mp-exp-item span{{color:#58a6ff;font-weight:600}}
  .mp-hist-title{{color:#8b949e;font-size:11px;margin-bottom:4px}}
  .mp-hist-list{{display:flex;gap:6px;flex-wrap:wrap}}
  .mp-hist-item{{font-size:11px;padding:2px 7px;border-radius:10px;
                 background:#21262d;color:#e6edf3}}
  .mp-hist-item .mtype{{color:#bc8cff;font-size:10px}}
  /* K线图 */
  .kline-wrap{{height:260px}}
  .sub-wrap{{height:100px;margin-top:4px}}
</style>
</head>
<body>
<header>
  <h1>📈 Sequoia-X 选股报告</h1>
  <p>生成时间：{today_str} &nbsp;|&nbsp; 共 {len(summary_rows)} 只股票
     {" &nbsp;|&nbsp; ⚡ 含形态匹配得分" if has_match else ""}
  </p>
</header>
<div class="container">

  <div class="summary-section">
    <div class="section-title">选股汇总</div>
    {summary_html}
  </div>

  <div class="charts-section">
    <div class="section-title">K线图 &amp; 指标{" &amp; 形态匹配" if has_match else ""}</div>
    <div id="charts-container"></div>
  </div>

</div>

<script>
const CHARTS_DATA = {charts_json};

const strategies = [...new Set(CHARTS_DATA.map(d => d.strategy))];
const container  = document.getElementById('charts-container');

strategies.forEach(strategy => {{
  const items = CHARTS_DATA.filter(d => d.strategy === strategy);
  const block = document.createElement('div');
  block.className = 'strategy-block';
  block.id = 'block-' + strategy;
  block.innerHTML = `<div class="strategy-block-title">${{strategy}}（${{items.length}} 只）</div>
                     <div class="charts-grid" id="grid-${{strategy}}"></div>`;
  container.appendChild(block);

  const grid = document.getElementById('grid-' + strategy);
  items.forEach(d => {{
    const pctClass = d.pct > 0 ? 'up' : d.pct < 0 ? 'down' : 'flat';
    const pctSign  = d.pct > 0 ? '+' : '';

    // 形态匹配面板 HTML
    let matchHtml = '';
    if (d.match_score !== null && d.match_score !== undefined) {{
      const scoreColor = d.match_score >= 0.7 ? '#ef232a' :
                         d.match_score >= 0.5 ? '#f0c05a' : '#8b949e';
      const histItems = (d.top_matches || []).map(m =>
        `<span class="mp-hist-item">
           ${{m.lib_symbol}} ${{m.lib_date}}
           <span class="mtype">[${{m.market_type}}]</span>
           <span style="color:#14b143">历史得分:${{(m.future_score*100).toFixed(0)}}%</span>
         </span>`
      ).join('');
      matchHtml = `
        <div class="match-panel">
          <div class="mp-title">⚡ 形态匹配分析</div>
          <div class="mp-scores">
            <div class="mp-score-item">
              <span class="label">综合相似度</span>
              <span class="value" style="color:${{scoreColor}}">${{(d.match_score*100).toFixed(1)}}%</span>
            </div>
          </div>
          <div class="mp-exp">
            <div class="mp-exp-item">5日突破概率 <span>${{(d.expected_5d*100).toFixed(0)}}%</span></div>
            <div class="mp-exp-item">30日预期涨幅 <span>${{(d.expected_30d*100).toFixed(0)}}%</span></div>
            <div class="mp-exp-item">90日预期涨幅 <span>${{(d.expected_90d*100).toFixed(0)}}%</span></div>
          </div>
          ${{histItems ? `<div class="mp-hist-title">Top 相似历史形态：</div>
          <div class="mp-hist-list">${{histItems}}</div>` : ''}}
        </div>`;
    }}

    const card = document.createElement('div');
    card.className = 'chart-card';
    card.id = 'card-' + d.symbol;
    card.innerHTML = `
      <div class="chart-header">
        <span class="sym">${{d.symbol}}</span>
        <span class="price">${{d.last_close}}</span>
        <span class="badge ${{pctClass}}">${{pctSign}}${{d.pct}}%</span>
        ${{d.match_score !== null && d.match_score !== undefined ?
          `<span class="ms-badge">⚡ 形态 ${{(d.match_score*100).toFixed(1)}}%</span>` : ''}}
      </div>
      ${{matchHtml}}
      <div class="kline-wrap" id="kline-${{d.symbol}}"></div>
      <div class="sub-wrap"   id="kdj-${{d.symbol}}"></div>
      <div class="sub-wrap"   id="brick-${{d.symbol}}"></div>
    `;
    grid.appendChild(card);

    const klineChart = echarts.init(document.getElementById('kline-' + d.symbol), 'dark');
    klineChart.setOption({{
      backgroundColor: 'transparent',
      grid: {{ top: 8, bottom: 24, left: 56, right: 12 }},
      xAxis: {{ type: 'category', data: d.dates, axisLabel: {{ fontSize: 10 }},
                boundaryGap: true, axisLine: {{ lineStyle: {{ color: '#30363d' }} }} }},
      yAxis: {{ type: 'value', scale: true, axisLabel: {{ fontSize: 10 }},
                splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'cross' }} }},
      series: [
        {{ type: 'candlestick', name: 'K线', data: d.kline,
           itemStyle: {{ color:'#ef232a', color0:'#14b143',
                         borderColor:'#ef232a', borderColor0:'#14b143' }} }},
        {{ type: 'line', name: '知行趋势线', data: d.trend,
           lineStyle: {{ color:'#f0c05a', width:1.5 }}, symbol:'none' }},
        {{ type: 'line', name: '知行多空线', data: d.bull,
           lineStyle: {{ color:'#58a6ff', width:1.5, type:'dashed' }}, symbol:'none' }},
      ]
    }});

    const kdjChart = echarts.init(document.getElementById('kdj-' + d.symbol), 'dark');
    kdjChart.setOption({{
      backgroundColor: 'transparent',
      grid: {{ top: 4, bottom: 20, left: 56, right: 12 }},
      xAxis: {{ type: 'category', data: d.dates, show: false }},
      yAxis: {{ type: 'value', scale: true, axisLabel: {{ fontSize: 9 }},
                splitLine: {{ lineStyle: {{ color: '#21262d' }} }},
                min: v => Math.min(v.min - 5, -10) }},
      tooltip: {{ trigger: 'axis' }},
      series: [{{ type:'line', name:'KDJ-J', data:d.kdj_j,
        lineStyle:{{ color:'#bc8cff', width:1.2 }}, symbol:'none',
        markLine:{{ silent:true,
          data:[{{ yAxis:13, lineStyle:{{ color:'#ef232a', type:'dashed', width:1 }} }}],
          label:{{ formatter:'J=13', fontSize:9 }} }} }}]
    }});

    const brickChart = echarts.init(document.getElementById('brick-' + d.symbol), 'dark');
    brickChart.setOption({{
      backgroundColor: 'transparent',
      grid: {{ top: 4, bottom: 20, left: 56, right: 12 }},
      xAxis: {{ type: 'category', data: d.dates, show: false }},
      yAxis: {{ type: 'value', scale: true, axisLabel: {{ fontSize: 9 }},
                splitLine: {{ lineStyle: {{ color: '#21262d' }} }} }},
      tooltip: {{ trigger: 'axis' }},
      series: [{{ type:'bar', name:'砖型图', data:d.brick,
        itemStyle:{{ color: params => d.brick_colors[params.dataIndex] }} }}]
    }});

    echarts.connect([klineChart, kdjChart, brickChart]);
  }});
}});

document.querySelectorAll('.chip[data-sym]').forEach(el => {{
  el.addEventListener('click', () => {{
    const card = document.getElementById('card-' + el.dataset.sym);
    if (card) card.scrollIntoView({{ behavior:'smooth', block:'center' }});
  }});
}});
</script>
</body>
</html>"""

    out = Path(output_path)
    out.write_text(html, encoding="utf-8")
    return str(out.resolve())
