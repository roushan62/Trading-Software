"""AI advisor module (OpenRouter + offline fallback)."""
from .advisor import (
    Advisor,
    build_market_context,
    get_api_key,
    local_advice,
    render_context,
    save_api_key,
    SYSTEM_PROMPT,
)

__all__ = [
    "Advisor", "build_market_context", "render_context", "local_advice",
    "get_api_key", "save_api_key", "SYSTEM_PROMPT",
]
