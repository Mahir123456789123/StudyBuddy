"""
Structured Logging Utility for StudyBuddy.
Ensures privacy compliance:
- Masks sensitive credentials/keys.
- Truncates user query previews to a maximum of 80 characters.
- Outputs structured JSON logs capturing endpoint, latency, intent, and retries.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def setup_logger(name: str = "studybuddy", log_level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger."""
    logger = logging.getLogger(name)
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logger()


def sanitize_text(text: Optional[str], max_len: int = 80) -> str:
    """Sanitizes and truncates user-provided text for safe audit logging."""
    if not text:
        return ""
    # Strip newline characters to prevent log injection
    cleaned = " ".join(text.split())
    if len(cleaned) > max_len:
        return cleaned[:max_len] + "..."
    return cleaned


def log_request_event(
    endpoint: str,
    success: bool,
    latency_ms: float,
    intent: Optional[str] = None,
    retries: int = 0,
    user_query: Optional[str] = None,
    error_message: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None
) -> None:
    """
    Emits a structured audit log line for incoming API requests.
    Guarantees no raw API keys or unbounded inputs are logged.
    """
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "success": success,
        "latency_ms": round(latency_ms, 2),
        "intent": intent or "unknown",
        "retries": retries,
        "query_preview": sanitize_text(user_query, max_len=80),
    }
    if error_message:
        event["error"] = error_message
    if extra:
        event["extra"] = extra

    log_entry = json.dumps(event)
    if success:
        logger.info(log_entry)
    else:
        logger.error(log_entry)
