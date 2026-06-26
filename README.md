# Sequoia-X

> A 股量化选股系统 | 知行 KDJ × 砖型图 × 形态匹配 × 超额收益（Alpha）评分

---

## 系统概览

Sequoia-X 是面向 A 股市场的量化选股系统，核心流程如下：

```
每日 15:35（交易日收盘后）
    ↓
增量拉取当日行情（baostock）
    ↓
运行选股策略（KDJ + 砖型图）
    ↓
形态匹配（DTW 粗筛 → SSIM 图像精排）
    ↓
超额收益评分（个股涨幅 - 沪深300涨幅）
    ↓
生成 HTML 报告 + 推送 Top10 到企业微信
```

---

## 快速开始

### 1. 安装依赖

```bash
# 推荐 uv（快速）
uv sync

# 或 pip
pip install .
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填写飞书 Webhook URL（不用飞书可填任意值）
```

### 3. 首次回填历史数据（约 12 分钟）

```bash
.venv/bin/python main.py --backfill
```

约 5200 只 A 股，2024-01-01 至今，后复权日 K 线。

### 4. 构建历史形态库（约 3 分钟）

```bash
.venv/bin/python run_zhixing_test.py --build-library
```

扫描所有股票近 300 天的信号日，提取 K 线形态并计算超额收益（Alpha）评分，入库约 20 万条形态。

### 5. 日常运行

```bash
# 选股 + 形态匹配 + HTML 报告（自动打开浏览器）
.venv/bin/python run_zhixing_test.py

# 选股 + 形态匹配 + 企业微信推送 Top10
WECOM_WEBHOOK_KEY=你的key .venv/bin/python run_zhixing_test.py --notify --no-open
```

---

## 定时任务（macOS）

项目内置 `com.sequoia-x.daily.plist`，每个交易日 15:35 自动运行并推送。

**安装步骤：**

```bash
# 1. 填写企业微信 Key
#    编辑 com.sequoia-x.daily.plist，替换 YOUR_WECOM_WEBHOOK_KEY_HERE

# 2. 安装定时任务
cp com.sequoia-x.daily.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.sequoia-x.daily.plist

# 3. 验证已注册
launchctl list | grep sequoia
```

日志输出到 `logs/sequoia_daily.log`。

---

## 数据存储

本项目使用 Git LFS 管理数据库文件，`git clone` 后数据即可用，无需重新回填。

| 文件 | 大小 | 内容 |
|------|------|------|
| `data/sequoia_v2.db` | ~393 MB | 全市场日 K 线（后复权，2024 至今） |
| `data/pattern_library.db` | ~400 MB | 历史形态库（20 万条，含 Alpha 评分） |
| `data/benchmark.db` | ~152 KB | 沪深300收盘价（2014 至今，用于计算超额收益） |

> 若 Git LFS 未安装：`brew install git-lfs && git lfs install && git lfs pull`

---

## 选股策略

| 策略 | 文件 | 触发条件 |
|------|------|----------|
| **知行 KDJ 低位+趋势** | `zhixing_kdj_trend.py` | J≤13 + 涨幅-2.5\%~+3\% + 振幅<7\% + 短期趋势线>多空线 |
| **知行砖型图红绿柱翻转** | `zhixing_brick_reversal.py` | 砖型图指标今日>昨日 + 昨日<前日 + 振幅放大 + 趋势向上 |

其他内置策略（暂未启用）：海龟突破、均线放量、高窄旗形、涨停洗盘、RPS 突破等。

---

## 形态匹配原理

**Step 1 — DTW 时序粗筛**
- 提取当前股票最近 30 根 K 线的 close 归一化序列
- 与形态库中随机采样的 5000 条历史形态做 DTW 距离计算
- 保留距离最小的 Top-20 候选

**Step 2 — SSIM 图像精排**
- 把候选形态和当前形态各自渲染为 128×64 K 线灰度图
- 用 SSIM（结构相似度）计算图像相似度
- `final_score = DTW × 0.4 + SSIM × 0.6`，取 Top-5

---

## 超额收益评分（Alpha）

形态库中每条记录的 `future_score` 基于超额收益计算，剔除牛市系统性涨幅：

```
Alpha = 个股涨幅 - 同期沪深300涨幅

score_5d  = 5日内突破信号前30日最高价（0 or 1）
score_30d = 30日 Alpha 线性映射：[-5%, +10%] → [0, 1]
score_90d = 90日 Alpha 线性映射：[-5%, +20%] → [0, 1]

future_score = 0.3 × score_5d + 0.4 × score_30d + 0.3 × score_90d
```

---

## 企业微信推送

推送内容：
1. **Markdown 摘要**：Top10 股票 + 编号/名称/板块/策略标签/砖型图🧱/形态匹配分/30日及90日超额收益预期
2. **HTML 附件**：`report_top10.html`，含砖型图柱状图，下载后浏览器打开

**配置方式：** 企业微信群 → 添加机器人 → 复制 Webhook URL 中的 `key=xxx` 部分

---

## 目录结构

```
Sequoia-X/
├── main.py                          # 入口：回填/增量数据同步
├── run_zhixing_test.py              # 主脚本：选股 → 匹配 → 报告 → 推送
├── com.sequoia-x.daily.plist        # macOS 定时任务配置
├── data/                            # SQLite 数据库（Git LFS 管理）
│   ├── sequoia_v2.db                # 主行情库
│   ├── pattern_library.db           # 历史形态库
│   └── benchmark.db                 # 沪深300基准
├── sequoia_x/
│   ├── core/
│   │   ├── config.py                # 配置管理（pydantic-settings）
│   │   └── logger.py                # 日志（rich）
│   ├── data/
│   │   └── engine.py                # 数据引擎（baostock + SQLite）
│   ├── pattern_library.py           # 形态库构建（Alpha 评分）
│   ├── pattern_matcher.py           # 形态匹配（DTW + SSIM）
│   ├── report.py                    # HTML 报告生成
│   ├── strategy/
│   │   ├── base.py                  # 策略基类
│   │   ├── zhixing_kdj_trend.py     # 知行 KDJ 策略
│   │   ├── zhixing_brick_reversal.py# 知行砖型图策略
│   │   └── ...                      # 其他内置策略
│   └── notify/
│       ├── feishu.py                # 飞书推送
│       └── wecom.py                 # 企业微信推送
└── logs/                            # 运行日志（定时任务输出）
```

---

## 环境变量

| 变量 | 必填 | 说明 |
|------|------|------|
| `FEISHU_WEBHOOK_URL` | 是 | 飞书机器人 Webhook URL（不用飞书填任意值） |
| `WECOM_WEBHOOK_KEY` | 否 | 企业微信群机器人 key（`--notify` 时必填） |

---

## 依赖要求

- Python >= 3.10
- 主要依赖：`baostock`、`pandas`、`numpy`、`matplotlib`、`Pillow`、`pydantic-settings`、`requests`
- 可选依赖：`fastdtw`（DTW 加速）、`scikit-image`（SSIM 精确计算）

---

## License

MIT
