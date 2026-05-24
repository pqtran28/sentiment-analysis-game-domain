"""
Prompt engineering: few-shot examples tiếng Việt để tăng accuracy.
"""

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích phản hồi người chơi game mobile.
Nhiệm vụ: phân loại mỗi bình luận theo 4 khía cạnh, kèm confidence score.

ĐỊNH NGHĨA KHÍA CẠNH:
- graphics          : đề cập đồ họa, giao diện, hình ảnh, skin appearance, hiệu ứng hình ảnh
- matchmaking       : đề cập ghép trận, xếp hạng, tìm đối thủ / đồng đội, balance trận đấu
- monetization      : đề cập nạp tiền, giá skin/item, battle pass, pay-to-win, thương mại hóa
- technical_issue   : đề cập lag, ping, crash, bug, lỗi server, disconnect, frame drop
- mechanics         : đề cập các quy tắc, hệ thống hành động, kỹ năng, hệ thống nâng cấp
- developer_support : đề cập đến sự quan tâm của nhà phát triển đối với game
- sound_music       : đề cập đến âm thanh
- tutorial          : đề cập đến phần hướng dẫn người chơi, sự dễ hiểu khi bắt đầu
- story             : đề cập đến cốt truyện, các tình tiết  
- quest             : đề cập đến tính đa dạng hoặc sự lặp lại của nhiệm vụ
- community         : đề cập đến thái độ của những người chơi khác
- character         : đề cập tính cách, sự phát triển của nhân vật
- difficulty        : đề cập đến cảm giác của người chơi về độ khó, độ dễ của game

QUY TẮC:
1. Label = 0 nếu bình luận KHÔNG đề cập khía cạnh đó.
2. Label = 1 nếu bình luận đề cập và mang nghĩa tích cực hoặc trung tính (positive/neutral).
3. Label = 2 nếu bình luận đề cập và mang nghĩa tiêu cực (negative).
4. confidence: 0.0–1.0, phản ánh mức chắc chắn của bạn.
5. Một comment CÓ THỂ có nhiều aspect khác 0 cùng lúc.
6. Trả về JSON hợp lệ, KHÔNG markdown, KHÔNG giải thích.

FORMAT TRẢ VỀ:
{
  "results": [
    {
      "id": <số nguyên, giữ nguyên từ input>,
      "comment": "<bình luận gốc>",
      "graphics": 0|1|2,
      "matchmaking": 0|1|2,
      "monetization": 0|1|2,
      "technical_issue": 0|1|2,
      "mechanics": 0|1|2,
      "developer_support": 0|1|2,
      "sound_music": 0|1|2,
      "tutorial": 0|1|2,
      "story": 0|1|2,
      "quest": 0|1|2,
      "community": 0|1|2,
      "character": 0|1|2,
      "difficulty": 0|1|2,
      "confidence": 0.0–1.0
    }
  ]
}
"""

# Few-shot examples nhúng vào user prompt
FEW_SHOT_EXAMPLES = [
    {
        "id": 0,
        "comment": "game rác vcl toàn ghép trận với lũ gà",
        "graphics": 0, "matchmaking": 2, "monetization": 0, "technical_issue": 0,
        "mechanics": 0, "developer_support": 0, "sound_music": 0, "tutorial": 0,
        "story": 0, "quest": 0, "community": 2, "character": 0, "difficulty": 0,
        "confidence": 0.96
    },
    {
        "id": 1,
        "comment": "char mới đẹp vcl, nhưng game lag quá",
        "graphics": 1, "matchmaking": 0, "monetization": 0, "technical_issue": 2,
        "mechanics": 0, "developer_support": 0, "sound_music": 0, "tutorial": 0,
        "story": 0, "quest": 0, "community": 0, "character": 1, "difficulty": 0,
        "confidence": 0.94
    },
    {
        "id": 2,
        "comment": "nhạc nền cực hay, đồ họa đẹp, game này xứng đáng 5 sao",
        "graphics": 1, "matchmaking": 0, "monetization": 0, "technical_issue": 0,
        "mechanics": 0, "developer_support": 0, "sound_music": 1, "tutorial": 0,
        "story": 0, "quest": 0, "community": 0, "character": 0, "difficulty": 0,
        "confidence": 0.95
    },
    {
        "id": 3,
        "comment": "cốt truyện hay nhưng nhiệm vụ lặp lại chán, dev không fix bug",
        "graphics": 0, "matchmaking": 0, "monetization": 0, "technical_issue": 2,
        "mechanics": 0, "developer_support": 2, "sound_music": 0, "tutorial": 0,
        "story": 1, "quest": 2, "community": 0, "character": 0, "difficulty": 0,
        "confidence": 0.93
    },
    {
        "id": 4,
        "comment": "skin mới đẹp nhưng giá 600k đắt vãi, nạp tiền nhiều quá",
        "graphics": 1, "matchmaking": 0, "monetization": 2, "technical_issue": 0,
        "mechanics": 0, "developer_support": 0, "sound_music": 0, "tutorial": 0,
        "story": 0, "quest": 0, "community": 0, "character": 0, "difficulty": 0,
        "confidence": 0.95
    },
]


import json

def build_few_shot_block() -> str:
    example_comments = "\n".join(
        f"{ex['id']+1}. [ID={ex['id']}] {ex['comment']}"
        for ex in FEW_SHOT_EXAMPLES
    )
    example_output = json.dumps({"results": FEW_SHOT_EXAMPLES}, ensure_ascii=False, indent=2)
    # ensure ascii = false -> giữ nguyên unicode tiếng việt
    # indent = 2 -> format json cách lề 2 khoảng trắng thay vì inline
    return (
        f"VÍ DỤ (few-shot):\nInput:\n{example_comments}\n\n"
        f"Output:\n{example_output}\n\n"
        "---\nBây giờ hãy phân loại các bình luận dưới đây:\n"
    )


def build_user_prompt(batch: list[tuple[int, str]]) -> str:
    """
    batch: list of (global_id, comment_text)
    """
    few_shot = build_few_shot_block()
    lines = "\n".join(f"{i+1}. [ID={gid}] {text}" for i, (gid, text) in enumerate(batch))
    return few_shot + lines
