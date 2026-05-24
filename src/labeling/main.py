"""
Entry point:
  python main.py              → chạy ngay
  python main.py --schedule   → lập lịch tự động (chạy nền, không cần mở máy nếu dùng cron/systemd)
  python main.py --reset      → xóa checkpoint, chạy lại từ đầu
  python main.py --review     → in ra các comment cần review thủ công
"""
import argparse
import logging
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)-14s] %(levelname)s – %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
    ],
)

# Thêm thư mục gốc vào path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import run_pipeline
from config.settings import (
    SCHEDULE_HOUR, SCHEDULE_MINUTE, OUTPUT_FILE,
)


def run_scheduled():
    """Chạy lặp theo lịch"""
    import schedule
    import time

    def job():
        logging.getLogger(__name__).info("⏰ Scheduled job bắt đầu...")
        run_pipeline()

    schedule.every().day.at(f"{SCHEDULE_HOUR:02d}:{SCHEDULE_MINUTE:02d}").do(job)
    logging.getLogger(__name__).info(
        "📅 Đã lập lịch: chạy mỗi ngày lúc %02d:%02d. Ctrl+C để dừng.",
        SCHEDULE_HOUR, SCHEDULE_MINUTE,
    )
    while True:
        schedule.run_pending()
        time.sleep(30)


def show_review_queue():
    """In các comment có confidence thấp cần review thủ công."""
    if not OUTPUT_FILE.exists():
        print("Chưa có file kết quả. Hãy chạy pipeline trước.")
        return
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    reviews = [r for r in data["results"] if r.get("needs_review")]
    if not reviews:
        print("✅ Không có comment nào cần review.")
        return
    print(f"\n⚠️  {len(reviews)} comment cần review thủ công:\n")
    for r in reviews:
        print(f"  ID {r['id']:>4} | conf={r['confidence']:.2f} | {r['comment'][:80]}")
        labels = {k: r[k] for k in ["graphics","matchmaking","monetization","technical_issue"]}
        print(f"           Labels: {labels}\n")


def main():
    Path("logs").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    parser = argparse.ArgumentParser(description="Gemini Aspect Classifier")
    parser.add_argument("--schedule", action="store_true", help="Lập lịch tự động")
    parser.add_argument("--reset",    action="store_true", help="Xóa checkpoint và chạy lại")
    parser.add_argument("--review",   action="store_true", help="Xem comment cần review")
    args = parser.parse_args()

    if args.review:
        show_review_queue()
    elif args.schedule:
        run_scheduled()
    else:
        run_pipeline(clear_checkpoint=args.reset)


if __name__ == "__main__":
    main()
