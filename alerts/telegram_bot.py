"""Telegram alerts via the Bot API (python-telegram-bot not required — a
plain requests call is enough and avoids an asyncio dependency).

Setup (2 minutes):
  1. In Telegram, message @BotFather -> /newbot -> copy the token.
  2. Message your new bot once (press START).
  3. Get your chat id: message @userinfobot (or use getUpdates below).
  4. Put token + chat id in config.json alerts section, set "telegram": true,
     or export TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import os

import requests

API_BASE = "https://api.telegram.org"


def send_telegram_message(
    text: str,
    bot_token: str | None = None,
    chat_id: str | None = None,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Send a text message. Returns (success, detail)."""
    token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = chat_id or os.environ.get("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        return False, "telegram not configured: missing TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID"
    try:
        resp = requests.post(
            f"{API_BASE}/bot{token}/sendMessage",
            json={"chat_id": chat, "text": text, "disable_web_page_preview": True},
            timeout=timeout,
        )
        if resp.status_code == 200 and resp.json().get("ok"):
            return True, "sent"
        return False, f"telegram API error {resp.status_code}: {resp.text[:200]}"
    except Exception as e:  # network down etc.
        return False, f"telegram send failed: {e}"


def get_chat_id(bot_token: str) -> str:
    """Helper: prints the latest incoming chat id (after you message the bot)."""
    resp = requests.get(f"{API_BASE}/bot{bot_token}/getUpdates", timeout=10)
    data = resp.json()
    for update in data.get("result", []):
        msg = update.get("message") or update.get("channel_post") or {}
        chat = msg.get("chat", {})
        if chat.get("id"):
            return str(chat["id"])
    return "(no messages yet — press START on your bot and retry)"
