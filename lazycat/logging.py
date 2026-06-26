"""
Centralized Log Manager and get_logger factory.
"""

import json
import logging
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

def get_logger(name: str) -> logging.Logger:
    """Factory to get a pre-configured structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        # In a real system, you'd use a JSONFormatter here, 
        # but for now we'll stick to a clean standard format
        formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

logger = get_logger(__name__)

class LogManager:
    BASE_DIR = Path("logs")
    CYCLE_DIR = BASE_DIR / "cycles"
    AB_DIR = CYCLE_DIR / "ab_results"

    def __init__(self):
        try:
            self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)
            test_file = self.CYCLE_DIR / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception:
            self.BASE_DIR = Path("logs_local")
            self.CYCLE_DIR = self.BASE_DIR / "cycles"
            self.AB_DIR = self.CYCLE_DIR / "ab_results"
            try:
                self.CYCLE_DIR.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass

        try:
            self.AB_DIR.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @staticmethod
    def _write_jsonl(path: Path, data: dict):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, default=str) + "\n")
        except Exception as e:
            logger.debug("[LogManager] Write failed for %s: %s", path.name, e)

    def _cycle_path(self, cycle_id: str) -> Path:
        return self.CYCLE_DIR / f"{cycle_id}.jsonl"

    def log_v2_cycle(self, cycle_id: str, step_name: str, payload: dict):
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "info",
            "step": step_name,
            "ticker": payload.get("ticker", ""),
            "payload": payload,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)

    def log_agent_turn(self, cycle_id, agent_name, turn_index, action_type, *, ticker="", content_preview="", tool_calls=None, tool_results=None, tokens_used=0, elapsed_ms=0, finish_reason="", extra=None):
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "agent_name": agent_name,
            "turn_index": turn_index,
            "action_type": action_type,
            "content_preview": content_preview[:500] if content_preview else "",
            "tokens_used": tokens_used,
            "elapsed_ms": elapsed_ms,
        }
        if finish_reason:
            payload["finish_reason"] = finish_reason
        if tool_calls:
            payload["tool_calls"] = [{"name": tc.get("function", {}).get("name", "?"), "args_preview": str(tc.get("function", {}).get("arguments", ""))[:300]} for tc in tool_calls[:10]]
        if tool_results:
            payload["tool_results"] = [{"name": tr.get("name", "?"), "content_preview": str(tr.get("content", ""))[:300]} for tr in tool_results[:10]]
        if extra:
            payload.update(extra)

        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "info" if action_type != "error" else "warning",
            "step": f"agent_turn_{action_type}",
            "ticker": ticker,
            "payload": payload,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)

    def log_truncation_warning(self, cycle_id, agent_name, ticker="", finish_reason="", response_preview=""):
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "warning",
            "step": "llm_truncation",
            "ticker": ticker,
            "payload": {
                "agent_name": agent_name,
                "finish_reason": finish_reason,
                "response_preview": response_preview[:500] if response_preview else "",
                "message": f"LLM output for {agent_name} was truncated."
            },
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)
        logger.warning("[LogManager] LLM TRUNCATION: %s/%s finish_reason=%s", agent_name, ticker, finish_reason)

    def log_cycle_error(self, cycle_id, error_type, *, ticker="", error="", stack_trace="", stage="", elapsed_ms=0, extra=None):
        ts = datetime.now(timezone.utc).isoformat()
        payload = {
            "error_type": error_type,
            "error": error[:2000] if error else "",
            "stage": stage,
            "elapsed_ms": elapsed_ms,
        }
        if stack_trace:
            payload["stack_trace"] = stack_trace[:4000]
        if extra:
            payload.update(extra)
        if ticker:
            payload["ticker"] = ticker

        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "error",
            "step": f"error_{error_type}",
            "ticker": ticker,
            "payload": payload,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)

    def log_cycle_summary(self, cycle_id: str, summary: dict):
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = {
            "cycle_id": cycle_id,
            "timestamp": ts,
            "level": "info",
            "step": "cycle_summary",
            "ticker": "",
            "payload": summary,
        }
        self._write_jsonl(self._cycle_path(cycle_id), log_entry)
        
        # In the SDK we do not write to the database directly. 
        # The calling service should handle DB persistence if required.

log_manager = LogManager()
