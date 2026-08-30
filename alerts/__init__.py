"""Alerts dispatcher: console + file + Telegram + desktop notification.

Every alert payload ALWAYS carries the disclaimer line.
"""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from strategy.base import Signal
from .telegram_bot import send_telegram_message

RUNTIME_DIR = Path(__file__).resolve().parent.parent / "data" / "runtime"
ALERTS_LOG = RUNTIME_DIR / "alerts.log"

DEFAULTS = {
    "console": True,
    "file": True,
    "telegram": False,
    "telegram_bot_token": "",
    "telegram_chat_id": "",
    "desktop": False,
}


class Notifier:
    def __init__(self, config: dict | None = None, journal=None):
        cfg = {**DEFAULTS, **(config or {})}
        self.console = bool(cfg["console"])
        self.file = bool(cfg["file"])
        self.telegram = bool(cfg["telegram"])
        self.token = cfg["telegram_bot_token"]
        self.chat_id = cfg["telegram_chat_id"]
        self.desktop = bool(cfg["desktop"])
        self.journal = journal

    # ------------------------------------------------------------------ #
    def send(self, text: str, title: str = "SIGNAL") -> dict[str, tuple[bool, str]]:
        """Dispatch one payload to every enabled channel. Returns per-channel results."""
        results: dict[str, tuple[bool, str]] = {}
        payload = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {title}\n{text}"

        if self.console:
            print(f"\n=== ALERT ({title}) ===\n{payload}\n", file=sys.stderr)
            results["console"] = (True, "printed")

        if self.file:
            try:
                ALERTS_LOG.parent.mkdir(parents=True, exist_ok=True)
                with open(ALERTS_LOG, "a", encoding="utf-8") as f:
                    f.write(payload + "\n" + "-" * 60 + "\n")
                results["file"] = (True, str(ALERTS_LOG))
            except Exception as e:
                results["file"] = (False, str(e))

        if self.telegram:
            ok, detail = send_telegram_message(text, self.token, self.chat_id)
            results["telegram"] = (ok, detail)

        if self.desktop:
            results["desktop"] = self._desktop_notify(title, text)

        if self.journal is not None:
            for channel, (ok, _d) in results.items():
                self.journal.log_alert(channel, text, ok)
        return results

    def send_signal(self, sig: Signal, source: str = "paper") -> dict[str, tuple[bool, str]]:
        title = f"{sig.direction} {sig.symbol} [{sig.timeframe}]"
        return self.send(sig.alert_text(), title=f"{source.upper()} SIGNAL: {title}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _desktop_notify(title: str, text: str) -> tuple[bool, str]:
        try:
            first_line = text.splitlines()[0]
            if _try(["notify-send", title, first_line]):
                return True, "notify-send"
            if _try(["osascript", "-e", f'display notification "{first_line}" with title "{title}"']):
                return True, "osascript"
            return False, "no desktop notifier available (console/file alerts still work)"
        except Exception as e:
            return False, str(e)


def _try(cmd: list[str]) -> bool:
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=5)
        return True
    except Exception:
        return False
