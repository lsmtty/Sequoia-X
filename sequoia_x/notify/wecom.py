"""企业微信通知模块：推送 Top10 选股 HTML 报告到企业微信群机器人。

企业微信群机器人 Webhook 接口说明：
  - POST https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=<KEY>
  - 支持 text / markdown / news（图文）/ file 等消息类型
  - markdown 不支持内嵌 HTML，改用 news 消息类型发图文卡片（含缩略图URL）
  - 如需完整 HTML 报告，先上传文件再发 file 消息

用法：
    from sequoia_x.notify.wecom import WecomNotifier
    notifier = WecomNotifier(webhook_key="<your_key>")
    notifier.send_top10(top10_stocks, match_scores, stock_info)
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import requests

from sequoia_x.core.logger import get_logger

logger = get_logger(__name__)

_WECOM_WEBHOOK_BASE = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
_WECOM_UPLOAD_BASE  = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type=file"


class WecomNotifier:
    """企业微信群机器人推送器。

    Args:
        webhook_key: 企业微信群机器人的 key（Webhook URL 中的 key 参数）。
                     也可以通过环境变量 WECOM_WEBHOOK_KEY 设置。
    """

    def __init__(self, webhook_key: str | None = None) -> None:
        self.key = webhook_key or os.environ.get("WECOM_WEBHOOK_KEY", "")
        if not self.key:
            logger.warning("WECOM_WEBHOOK_KEY 未配置，企业微信推送将跳过")

    @property
    def _send_url(self) -> str:
        return _WECOM_WEBHOOK_BASE.format(key=self.key)

    @property
    def _upload_url(self) -> str:
        return _WECOM_UPLOAD_BASE.format(key=self.key)

    # ── 底层发送 ──────────────────────────────────────────────────────────

    def _post(self, payload: dict) -> bool:
        if not self.key:
            return False
        try:
            resp = requests.post(
                self._send_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=15,
            )
            data = resp.json()
            if data.get("errcode") == 0:
                logger.info("企业微信推送成功")
                return True
            else:
                logger.error(f"企业微信推送失败: {data}")
                return False
        except Exception as exc:
            logger.error(f"企业微信推送异常: {exc}")
            return False

    # ── 上传 HTML 文件 ────────────────────────────────────────────────────

    def _upload_file(self, file_path: str) -> str | None:
        """上传文件到企业微信，返回 media_id。"""
        if not self.key:
            return None
        try:
            with open(file_path, "rb") as f:
                fname = os.path.basename(file_path)
                resp = requests.post(
                    self._upload_url,
                    files={"media": (fname, f, "text/html")},
                    timeout=60,
                )
            data = resp.json()
            if data.get("errcode") == 0:
                media_id = data.get("media_id", "")
                logger.info(f"文件上传成功，media_id={media_id}")
                return media_id
            else:
                logger.error(f"文件上传失败: {data}")
                return None
        except Exception as exc:
            logger.error(f"文件上传异常: {exc}")
            return None

    # ── 发送 Markdown 摘要消息 ────────────────────────────────────────────

    def send_markdown(self, content: str) -> bool:
        """发送 markdown 消息（企业微信支持有限的 Markdown 语法）。"""
        return self._post({"msgtype": "markdown", "markdown": {"content": content}})

    # ── 发送文件消息 ──────────────────────────────────────────────────────

    def send_file(self, media_id: str) -> bool:
        """发送已上传的文件。"""
        return self._post({"msgtype": "file", "file": {"media_id": media_id}})

    # ── 核心：发送 Top10 报告 ─────────────────────────────────────────────

    def send_top10(
        self,
        top10: list[dict],
        report_html_path: str | None = None,
    ) -> None:
        """发送 Top10 选股摘要（Markdown）+ HTML 报告文件。

        Args:
            top10: Top10 股票信息列表，每条包含：
                   symbol, name, sector, strategy,
                   match_score, expected_30d, expected_90d,
                   brick_signal（砖型图是否触发）
            report_html_path: 完整 HTML 报告路径，若提供则上传并发送文件消息。
        """
        today = date.today().strftime("%Y-%m-%d")

        # 构建 Markdown 摘要
        lines = [
            f"## 📈 Sequoia-X 每日选股 Top10",
            f"> 日期：{today}",
            "",
        ]

        for i, s in enumerate(top10, 1):
            name    = s.get("name", s["symbol"])
            sector  = s.get("sector", "未知板块")
            score   = s.get("match_score", 0.0) or 0.0
            exp30   = s.get("expected_30d", 0.0) or 0.0
            exp90   = s.get("expected_90d", 0.0) or 0.0
            brick   = s.get("brick_signal", False)
            strategy = s.get("strategy", "")

            # 砖型图标记
            brick_tag = "🧱" if brick else "📊"

            # 分数颜色（企业微信 markdown 支持 <font color=> 标签）
            score_pct = int(score * 100)
            exp30_pct = int(exp30 * 100)
            exp90_pct = int(exp90 * 100)

            color = "warning" if score >= 0.6 else "comment"  # orange or grey

            lines.append(
                f"**{i}. {s['symbol']} {name}** {brick_tag}"
            )
            lines.append(
                f"> <font color=\"{color}\">形态匹配 {score_pct}%</font>　"
                f"板块：{sector}　策略：{strategy}"
            )
            lines.append(
                f"> 30日超额预期 <font color=\"info\">{exp30_pct}%</font>　"
                f"90日超额预期 <font color=\"info\">{exp90_pct}%</font>"
            )
            lines.append("")

        lines.append("> *完整 K 线报告见附件*")
        md_content = "\n".join(lines)

        # 发送 Markdown 摘要
        self.send_markdown(md_content)

        # 发送 HTML 报告文件
        if report_html_path and os.path.exists(report_html_path):
            media_id = self._upload_file(report_html_path)
            if media_id:
                self.send_file(media_id)
            else:
                logger.warning("HTML 文件上传失败，跳过文件推送")
