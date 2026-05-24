# Gemini Aspect-Based Comment Classifier

Phân loại bình luận game theo 4 khía cạnh: **graphics · matchmaking · monetization · technical_issue**

## Cấu trúc project

```
classifier/
├── main.py               ← entry point
├── pipeline.py           ← orchestrator đa luồng
├── gemini_caller.py      ← gọi Gemini API (realtime + batch)
├── key_pool.py           ← quản lý API keys, auto-rotate khi hết quota
├── checkpoint.py         ← lưu tiến độ, resume sau crash
├── config/
│   ├── settings.py       ← tất cả cấu hình (chỉnh ở đây)
│   └── prompts.py        ← system prompt + few-shot examples
├── data/
│   └── comments.csv      ← file input của bạn
├── output/
│   └── results.json      ← kết quả phân loại
├── logs/
│   └── pipeline.log
├── checkpoint.json        ← tự tạo khi chạy
├── requirements.txt
└── .env
```

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
# Điền API key thật vào .env
```

## Input

Đặt file comment vào `data/`, hỗ trợ:
- **CSV**: cần cột `comment` (đổi tên trong `config/settings.py` → `COMMENT_COLUMN`)
- **JSON**: `["comment1", "comment2"]` hoặc `[{"comment": "..."}]`
- **TXT**: mỗi dòng 1 comment

Đổi đường dẫn tại `config/settings.py` → `INPUT_FILE`.

## Chạy

```bash
# Chạy ngay
python main.py

# Chạy lại từ đầu (xóa checkpoint)
python main.py --reset

# Xem comment cần review thủ công (confidence thấp)
python main.py --review

# Lập lịch tự động (giữ terminal mở, hoặc dùng cron bên dưới)
python main.py --schedule
```

## Lập lịch không cần mở máy (Linux / Mac / WSL)

```bash
# Mở crontab
crontab -e

# Thêm dòng này → chạy lúc 2:00 AM mỗi ngày
0 2 * * * cd /đường/dẫn/đến/classifier && python main.py >> logs/cron.log 2>&1
```

### Windows (Task Scheduler)

1. Mở **Task Scheduler** → Create Basic Task
2. Trigger: Daily, lúc 2:00 AM
3. Action: Start a Program
   - Program: `python`
   - Arguments: `main.py`
   - Start in: `C:\đường\dẫn\đến\classifier`

## Output

File `output/results.json`:

```json
{
  "metadata": {
    "total": 1500,
    "needs_review": 42,
    "generated_at": "2025-08-01T02:15:30",
    "aspects": ["graphics", "matchmaking", "monetization", "technical_issue"]
  },
  "results": [
    {
      "id": 0,
      "comment": "game rác vcl toàn ghép trận với lũ gà",
      "graphics": 0,
      "matchmaking": 1,
      "monetization": 0,
      "technical_issue": 0,
      "confidence": 0.97,
      "needs_review": false
    }
  ]
}
```

## Tối ưu cho Free Tier (15 RPM / 1500 RPD)

| Setting | Giá trị gợi ý |
|---|---|
| `BATCH_SIZE` | 10 comments / request |
| `REQUESTS_PER_MINUTE` | 12 (buffer dưới 15) |
| `MODEL_NAME` | `gemini-1.5-flash` |

Với 2000 comments + batch 10 = **200 requests** → chạy xong trong ~17 phút.
Nếu có nhiều key thì nhanh hơn tỉ lệ thuận.

## Xử lý hết quota

Pipeline tự động:
1. Phát hiện lỗi quota (HTTP 429 / `RESOURCE_EXHAUSTED`)
2. Loại key đó khỏi pool active
3. Kéo key dự phòng từ `GEMINI_RESERVE_KEYS` vào thay thế
4. Tiếp tục không bị ngắt

## Review thủ công

Comment có `confidence < 0.75` sẽ có `"needs_review": true`.
Chạy `python main.py --review` để xem danh sách cần kiểm tra.
