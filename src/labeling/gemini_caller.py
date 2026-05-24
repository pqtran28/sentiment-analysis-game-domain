"""
Gemini caller:
  - Realtime mode : gọi trực tiếp, có rate-limit throttle + retry + key rotation
  - Batch API mode: submit file → poll → download (không cần mở máy)
"""
import json
import time
import logging
import threading
from pathlib import Path

import google.generativeai as genai

from config.prompts import SYSTEM_PROMPT, build_user_prompt
from config.settings import (
    MODEL_NAME, BATCH_SIZE, REQUESTS_PER_MINUTE,
    MAX_RETRIES, RETRY_BASE_DELAY, CONFIDENCE_THRESHOLD,
)
from key_pool import ApiKeyPool

log = logging.getLogger(__name__)

# ── Rate limiter (token bucket) ───────────────────────────────────────────────
class _RateLimiter:
    def __init__(self, rpm: int):
        self._interval = 60.0 / rpm
        self._lock     = threading.Lock()
        self._last     = 0.0

    def wait(self):
        with self._lock:
            now    = time.monotonic()
            sleep  = self._interval - (now - self._last)
            if sleep > 0:
                time.sleep(sleep)
            self._last = time.monotonic()

_limiter = _RateLimiter(REQUESTS_PER_MINUTE)


# ── Parse response ────────────────────────────────────────────────────────────
def _parse_response(text: str, batch_ids: list[int]) -> list[dict]:
    """Parse JSON từ model, map lại theo ID gốc."""
    clean = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
    data  = json.loads(clean)
    items = data.get("results", [])

    # Đảm bảo ID map đúng — dùng id từ response nếu có, fallback vào thứ tự
    parsed = []
    for i, item in enumerate(items):
        rid = item.get("id", batch_ids[i] if i < len(batch_ids) else -1)
        confidence = float(item.get("confidence", 1.0))
        parsed.append({
                "id":               rid,
                "comment":          item.get("comment", ""),
                "graphics":         int(item.get("graphics", 0)),
                "matchmaking":      int(item.get("matchmaking", 0)),
                "monetization":     int(item.get("monetization", 0)),
                "technical_issue":  int(item.get("technical_issue", 0)),
                "mechanics":        int(item.get("mechanics", 0)),       
                "developer_support":int(item.get("developer_support", 0)),
                "sound_music":      int(item.get("sound_music", 0)),      
                "tutorial":         int(item.get("tutorial", 0)),         
                "story":            int(item.get("story", 0)),            
                "quest":            int(item.get("quest", 0)),           
                "community":        int(item.get("community", 0)),        
                "character":        int(item.get("character", 0)),        
                "difficulty":       int(item.get("difficulty", 0)),      
                "confidence":       confidence,
                "needs_review":     confidence < CONFIDENCE_THRESHOLD,
            })
    return parsed


# ── Realtime call ─────────────────────────────────────────────────────────────
def call_realtime(
    batch: list[tuple[int, str]],
    key_pool: ApiKeyPool,
    thread_idx: int = 0,
) -> list[dict]:
    """
    Gọi Gemini realtime cho 1 batch.
    Tự đổi key nếu quota error, retry với exponential back-off.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        key = key_pool.get(thread_idx)
        if key is None:
            raise RuntimeError("Không còn API key khả dụng.")

        try:
            _limiter.wait()
            genai.configure(api_key=key)
            model = genai.GenerativeModel(
                model_name=MODEL_NAME,
                system_instruction=SYSTEM_PROMPT,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            response = model.generate_content(build_user_prompt(batch))
            batch_ids = [gid for gid, _ in batch]
            return _parse_response(response.text, batch_ids)

        except Exception as exc:
            if key_pool.is_quota_error(exc):
                key_pool.mark_quota_exhausted(key)
                log.warning("Quota hết, đổi key và thử lại ngay.")
                continue  # thử lại với key mới, không tính vào attempt
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            log.warning("Attempt %d/%d lỗi: %s. Retry sau %ds.", attempt, MAX_RETRIES, exc, delay)
            if attempt < MAX_RETRIES:
                time.sleep(delay)
            else:
                raise

# CÁI NÀY KO XÀI Á NHA
# ── Batch API mode ────────────────────────────────────────────────────────────
def submit_batch_job(
    all_comments: list[tuple[int, str]],
    api_key: str,
    job_state_file: Path,
) -> str:
    """
    Submit toàn bộ dataset lên Gemini Batch API.
    Trả về batch_job_name để poll sau.
    Lưu job name vào file để có thể poll lại sau khi tắt máy.
    """
    genai.configure(api_key=api_key)

    # Tạo JSONL requests
    requests_jsonl = []
    for i in range(0, len(all_comments), BATCH_SIZE):
        batch = all_comments[i : i + BATCH_SIZE]
        batch_ids = [gid for gid, _ in batch]
        req = {
            "custom_id": f"batch_{i}_{batch_ids[0]}_{batch_ids[-1]}",
            "request": {
                "model": MODEL_NAME,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_user_prompt(batch)}],
                "max_tokens": 2048,
            },
        }
        requests_jsonl.append(json.dumps(req, ensure_ascii=False))

    jsonl_content = "\n".join(requests_jsonl).encode("utf-8")

    # Upload file
    log.info("📤 Uploading %d requests lên Gemini Batch API...", len(requests_jsonl))
    uploaded = genai.upload_file(
        path=None,
        display_name="aspect_classification_batch",
        mime_type="application/jsonl",
    )
    # Note: API thực tế dùng files.create với content; đây là pseudocode cấu trúc
    # Xem docs: https://ai.google.dev/api/batch

    log.info("✅ Batch job submitted. Job name: %s", uploaded.name)

    # Lưu để poll sau
    job_state_file.parent.mkdir(parents=True, exist_ok=True)
    job_state_file.write_text(json.dumps({"job_name": uploaded.name, "total": len(all_comments)}))

    return uploaded.name


def poll_batch_job(job_name: str, api_key: str, poll_interval: int = 300) -> list[dict]:
    """
    Poll cho đến khi batch job hoàn thành (~24h).
    poll_interval: giây giữa các lần check (mặc định 5 phút).
    """
    genai.configure(api_key=api_key)
    log.info("⏳ Bắt đầu poll job: %s (interval=%ds)", job_name, poll_interval)

    while True:
        try:
            job = genai.get_batch_job(job_name)  # placeholder — xem SDK docs
            state = job.state.name
            log.info("Job state: %s", state)

            if state == "JOB_STATE_SUCCEEDED":
                log.info("✅ Batch job hoàn thành!")
                return _download_batch_results(job, api_key)
            elif state in ("JOB_STATE_FAILED", "JOB_STATE_CANCELLED"):
                raise RuntimeError(f"Batch job {state}: {job.error}")
            else:
                log.info("⏳ Đang xử lý... thử lại sau %ds.", poll_interval)
                time.sleep(poll_interval)
        except Exception as exc:
            if "not found" in str(exc).lower():
                raise
            log.warning("Poll error: %s. Retry sau 60s.", exc)
            time.sleep(60)


def _download_batch_results(job, api_key: str) -> list[dict]:
    """Download và parse kết quả từ batch job."""
    genai.configure(api_key=api_key)
    results = []
    # Giả lập download — thực tế dùng job.output_file hoặc tương đương
    for response in job.responses:
        try:
            items = _parse_response(response.content, [])
            results.extend(items)
        except Exception as e:
            log.warning("Parse lỗi 1 response: %s", e)
    return results
