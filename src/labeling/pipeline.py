"""
Pipeline chính:
- Load data → resume từ checkpoint
- Đa luồng (1 thread / key) với rate limiting
- Lưu checkpoint sau mỗi batch
- Xuất JSON kết quả cuối
"""
import os
import json
import queue
import threading
import logging
from pathlib import Path
from datetime import datetime

from config.settings import (
    INPUT_FILE, OUTPUT_FILE, CHECKPOINT_FILE,
    COMMENT_COLUMN, BATCH_SIZE, ASPECTS,
)
from key_pool  import ApiKeyPool
from checkpoint import CheckpointManager
from gemini_caller import call_realtime

log = logging.getLogger(__name__)


# ── Data loader ───────────────────────────────────────────────────────────────
def load_comments(path: Path) -> list[tuple[int, str]]:
    """Load comments từ CSV / JSON / TXT, trả về list (id, text)."""
    suffix = path.suffix.lower()

    if suffix == ".csv":
        import csv
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return [(i, row[COMMENT_COLUMN].strip()) for i, row in enumerate(reader) if row.get(COMMENT_COLUMN, "").strip()]

    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if data and isinstance(data[0], str):
                return [(i, s.strip()) for i, s in enumerate(data) if s.strip()]
            else:  # list of dicts
                return [(i, row[COMMENT_COLUMN].strip()) for i, row in enumerate(data) if row.get(COMMENT_COLUMN, "").strip()]

    elif suffix == ".txt":
        lines = path.read_text(encoding="utf-8").splitlines()
        return [(i, l.strip()) for i, l in enumerate(lines) if l.strip()]

    raise ValueError(f"Định dạng file không hỗ trợ: {suffix}")


# ── Writer ────────────────────────────────────────────────────────────────────
def save_results(results: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Sắp xếp theo ID gốc
    sorted_r = sorted(results, key=lambda x: x["id"])
    output = {
        "metadata": {
            "total":        len(sorted_r),
            "needs_review": sum(1 for r in sorted_r if r.get("needs_review")),
            "generated_at": datetime.now().isoformat(),
            "aspects":      ASPECTS,
        },
        "results": sorted_r,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log.info("✅ Đã lưu %d kết quả → %s", len(sorted_r), path)


# ── Worker thread ─────────────────────────────────────────────────────────────
def _worker(
    thread_idx: int,
    task_q: queue.Queue,
    key_pool: ApiKeyPool,
    ckpt: CheckpointManager,
    lock: threading.Lock,
):
    while True:
        try:
            batch: list[tuple[int, str]] = task_q.get(block=False)
        except queue.Empty:
            break

        try:
            results = call_realtime(batch, key_pool, thread_idx)
            with lock:
                ckpt.add_results(results)
            log.info("Thread-%d ✓ batch %d comments (ids %d–%d)",
                     thread_idx, len(batch), batch[0][0], batch[-1][0])
        except Exception as exc:
            log.error("Thread-%d ✗ batch thất bại: %s", thread_idx, exc)
        finally:
            task_q.task_done()


# ── Main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(clear_checkpoint: bool = False):
    # 1. Keys từ env
    primary_keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    reserve_keys = [k.strip() for k in os.getenv("GEMINI_RESERVE_KEYS", "").split(",") if k.strip()]
    if not primary_keys:
        raise ValueError("Thiếu GEMINI_API_KEYS trong .env")

    key_pool = ApiKeyPool(primary_keys, reserve_keys)
    ckpt     = CheckpointManager(CHECKPOINT_FILE)

    if clear_checkpoint:
        ckpt.clear()

    # 2. Load data
    log.info("📂 Đọc file: %s", INPUT_FILE)
    all_comments = load_comments(INPUT_FILE)
    log.info("   Tổng: %d comments.", len(all_comments))

    # 3. Lọc những cái đã xong
    done = ckpt.done_ids()
    remaining = [(gid, text) for gid, text in all_comments if gid not in done]
    log.info("   Còn lại: %d (bỏ qua %d đã có trong checkpoint).", len(remaining), len(done))

    if not remaining:
        log.info("✅ Tất cả đã được xử lý. Xuất kết quả...")
        save_results(ckpt.all_results(), OUTPUT_FILE)
        return

    # 4. Tạo batches → queue
    task_q: queue.Queue = queue.Queue()
    for i in range(0, len(remaining), BATCH_SIZE):
        task_q.put(remaining[i : i + BATCH_SIZE])
    total_batches = task_q.qsize()
    log.info("   %d batches (size=%d) → %d threads.", total_batches, BATCH_SIZE, len(primary_keys))

    # 5. Spawn threads
    lock    = threading.Lock()
    threads = []
    for idx in range(len(primary_keys)):
        t = threading.Thread(
            target=_worker,
            args=(idx, task_q, key_pool, ckpt, lock),
            name=f"Worker-{idx}",
            daemon=True,
        )
        threads.append(t)
        t.start()

    task_q.join()
    for t in threads:
        t.join()

    # 6. Xuất JSON
    save_results(ckpt.all_results(), OUTPUT_FILE)
    log.info("🏁 Pipeline hoàn thành. Key pool: %s", key_pool.status())
