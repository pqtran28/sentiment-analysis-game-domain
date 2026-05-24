"""
Toàn bộ cấu hình pipeline
"""
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
DATA_DIR        = BASE_DIR / "data"
OUTPUT_DIR      = BASE_DIR / "output"
LOG_DIR         = BASE_DIR / "logs"
CHECKPOINT_FILE = BASE_DIR / "checkpoint_bangbang.json"

# Input: file CSV / JSON / TXT chứa comments
# CSV: phải có cột tên theo COMMENT_COLUMN
# JSON: list of strings  hoặc list of objects
# TXT: mỗi dòng 1 comment
INPUT_FILE      = DATA_DIR / "lienquan_comments.csv"
COMMENT_COLUMN  = "comment"          # tên cột nếu dùng CSV / JSON object

OUTPUT_FILE     = OUTPUT_DIR / "lienquan_results.json"

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME      = "gemini-3.1-flash-lite-preview"  
USE_BATCH_API   = True                 # True = dùng Batch API (submit & poll, ~24h)
                                       # False = streaming realtime (cần mở máy)

# ── Rate limiting (free tier: 15 RPM, 1500 RPD) ───────────────────────────────
BATCH_SIZE          = 10     # số comment / 1 request
REQUESTS_PER_MINUTE = 10      
MAX_RETRIES         = 5
RETRY_BASE_DELAY    = 4      # seconds, exponential back-off

# ── Confidence & human review ─────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.75  # dưới ngưỡng này → đánh dấu needs_review=True
REQUEST_CONFIDENCE   = True  # yêu cầu model trả thêm confidence score

# ── Scheduling (cron-style, chạy tự động) ────────────────────────────────────
# Dùng khi set SCHEDULE_ENABLED=True trong .env
SCHEDULE_HOUR   = 2    # chạy lúc 2:00 AM mỗi ngày
SCHEDULE_MINUTE = 0

# ── Aspects ───────────────────────────────────────────────────────────────────
ASPECTS = [
    "graphics", 
    "matchmaking", 
    "monetization", 
    "technical_issue", 
    "mechanics", 
    "developer_support", 
    "sound_music", 
    "tutorial", 
    "story", 
    "quest", 
    "community", 
    "character", 
    "difficulty"
]