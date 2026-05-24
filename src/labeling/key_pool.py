"""
Thread-safe API key pool: primary keys + reserve keys, tự động xoay khi hết quota.
"""
import threading
import logging

log = logging.getLogger(__name__)

QUOTA_SIGNALS = ("quota", "429", "resource_exhausted", "rate_limit", "too many requests")


class ApiKeyPool:
    def __init__(self, primary: list[str], reserve: list[str]):
        self._lock    = threading.Lock()
        self._active  = list(primary) # các API key đang dùng được
        self._reserve = list(reserve) # các API key dự phòng để xoay
        self._bad: set[str] = set() # các API key đã hết quota

    # ── public ───────────────────────────────────────────────────────────────

    def get(self, thread_idx: int = 0) -> str | None:
        with self._lock:
            if not self._active:
                return None
            return self._active[thread_idx % len(self._active)]

    def mark_quota_exhausted(self, key: str):
        with self._lock:
            if key in self._active:
                self._active.remove(key)
                self._bad.add(key)
                log.warning("🔑 Key ...%s hết quota → loại khỏi pool active.", key[-8:])
            if self._reserve:
                promoted = self._reserve.pop(0)
                self._active.append(promoted)
                log.info("🔑 Key dự phòng ...%s → đưa vào pool active.", promoted[-8:])
            if not self._active:
                log.critical("❌ Không còn API key nào khả dụng!")

    def is_quota_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in QUOTA_SIGNALS)

    def available(self) -> bool:
        with self._lock:
            return bool(self._active)

    def status(self) -> dict:
        with self._lock:
            return {
                "active": len(self._active),
                "reserve": len(self._reserve),
                "exhausted": len(self._bad),
            }
