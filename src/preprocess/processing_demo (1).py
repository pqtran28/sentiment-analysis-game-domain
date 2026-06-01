import os
import time
import re
import json
import pandas as pd
from langdetect import detect 
from google import genai
from google.genai.errors import APIError
import unicodedata
from pyvi import ViTokenizer
from underthesea import word_tokenize
from dotenv import load_dotenv

load_dotenv()
class KeyManager:
    def __init__(self):
        self.keys = []
        i = 1
        while True:
            key = os.environ.get(f"GENAI_API_KEY_{i}")
            if not key:
                break
            self.keys.append(key)
            i += 1
        if not self.keys:
            raise ValueError("Khong tim thay API Key")
        self.current_index = 0
        self.cooldown_until = {}  # thoi gian key het cooldown
        print(f"Tong key da load: {len(self.keys)} key")

    # Ham cahy client voi key hien tai
    def get_client(self):
        while True:
            now = time.time()
            for _ in range(len(self.keys)):
                key = self.keys[self.current_index]
                cooldown_end = self.cooldown_until.get(key, 0)
                if now >= cooldown_end:
                    return genai.Client(api_key=key), key
                print(f"Key {key[-6:]} đang cooldown, chuyen qua key khac.")
                self.current_index = (self.current_index + 1) % len(self.keys)
            # neu tat ca key cooldown -> cho key gan nhat het cooldown
            nearest_key = min(self.cooldown_until, key=self.cooldown_until.get)
            wait_time = self.cooldown_until[nearest_key] - now + 1
            print(f"Tat ca key đang cooldown. Cho {wait_time:.1f} giay")
            time.sleep(wait_time)

    # Ham danh dau key bi loi 429 (rate limit) va doi qua key tiep theo
    def mark_rate_limited(self, key, cooldown_seconds=60):
        self.cooldown_until[key] = time.time() + cooldown_seconds
        print(f"Key {key[-6:]} bi rate limit. Cooldown {cooldown_seconds} giay.Doi qua key tiep theo.")
        self.current_index = (self.current_index + 1) % len(self.keys)

key_manager = KeyManager()

RETRY_BASE_DELAY = 4 # seconds, exponential back-off
MAX_RETRIES = 5
REQUESTS_PER_MINUTE = 10
# rate limiting
_min_interval = 60.0 / REQUESTS_PER_MINUTE
CACHE_FILE = "cache.json"

# load nhung review da xu ly trong cache
def load_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}
CACHE = load_cache() 

# luu nhung review da xu ly
def save_cache():
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Khong the luu cache: {e}")

# Ham goi API voi co che Exponential Backoff de xu ly loi, co retry neu that bai
def api_with_retry(prompt, config, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        client, current_key = key_manager.get_client()
        try:
            time.sleep(_min_interval) # sleep 6s truoc moi request
            response = client.models.generate_content(
                model = "gemini-3.1-flash-lite-preview",
                contents = prompt,
                config = config
            )
            # xoa khoang trang thua o dau va cuoi response
            response = response.text.strip()
            return response
        except APIError as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            print(f"Loi API lan {attempt+1}: {e}")
            if "Rate limit exceeded" in str(e) or "429" in str(e):
                key_manager.mark_rate_limited(current_key, cooldown_seconds=60)
                if attempt < max_retries - 1:
                    print(f"Loi Rate-limit lan thu {attempt + 1}/{max_retries} lan. Thu lai voi key moi sau {wait} giay")
                else:
                    print(f"That bai {max_retries} lan do Rate-limit. Ket thuc qua trinh.")
        except Exception as e:
            wait = RETRY_BASE_DELAY * (2 ** attempt)
            if attempt < max_retries - 1:
                print(f"Loi khong xac dinh. Thu lai lan {attempt + 1}/{max_retries} lan sau {2 ** attempt} giay...")
                time.sleep(wait)
            else:
                print(f"That bai {max_retries} lan do loi khong xac dinh. Ket thuc qua trinh.")
    return None

def normalize_with_gemini(text_input) -> str:
    system_instruction = (
        "Bạn là một chuyên gia tiền xử lý ngôn ngữ tự nhiên (NLP Text Preprocessing), am hiểu ngôn ngữ mạng và thuật ngữ game MOBA."
        "Nhiệm vụ của bạn là chuẩn hóa các đoạn review game (từ 1 sao đến 3 sao) để làm đầu vào cho mô hình học máy Phân loại cảm xúc theo khía cạnh (ABSA). "
        "Tuyệt đối tuân thủ các quy tắc sau:"
        "1. BẢO TỒN CẤU TRÚC VÀ Ý NGHĨA: Tuyệt đối không tự ý viết lại câu, không tóm tắt, không đảo trật tự từ. Chỉ thay thế/sửa từ ngay tại vị trí của nó."
        "2. Sửa lỗi chính tả và viết tắt thông dụng: 'ko/k' -> 'không', 'dc' -> 'được', 'nv' -> 'nhân vật', 'nph' -> 'nhà phát hành', 'acc' -> 'tài khoản', 'cs' -> 'có', 'j' -> 'gì', 'cx' -> 'cũng', 's' -> 'sao', 'r' -> 'rồi', 'lỏ' -> 'lỗi', 'chs' -> 'chơi', 'v' -> 'vậy'."
        "3. Bảo tồn THUẬT NGỮ GAME và KHÍA CẠNH: Tuyệt đối giữ nguyên tiếng Anh và các từ chỉ khía cạnh game: lag, bug, ping, fps, p2w, pay to win, nerf, buff, meta, dev, hack, troll, feed, ks, ad, sp, ap, top, mid, rừng, bot, skin, tank, gank, combat, backdoor, combo, farm, rank, elo, ghép trận, cày thuê, đồ họa, âm thanh, nạp thẻ, sự kiện, tướng."
        "4. Chuẩn hóa từ ngữ thô tục/nhấn mạnh: "
        "   - Chuyển các từ chửi thề (đm, rác rưởi, óc chó, súc vật, cc, l...) thành 'tệ hại'. "
        "     LƯU Ý: Không được xóa danh từ đứng trước hoặc sau nó. "
        "     VD: 'game rác rưởi' -> 'game tệ hại', 'hay như cc' -> 'tệ hại', 'game l' -> 'game tệ hại'."
        "   - Các từ lóng nhấn mạnh (vcl, vl, loz) phải đổi thành 'vô cùng'. "
        "     VD: 'lag vcl' -> 'lag vô cùng', 'hay vl' -> 'hay vô cùng'."
        "5. Dấu câu và định dạng: Giữ nguyên các dấu chấm (.), phẩy (,), chấm than (!) và hỏi chấm (?). Bỏ các icon/emoji. Chuyển toàn bộ văn bản thành chữ thường."
        "6. Trả về DUY NHẤT văn bản đã chuẩn hóa, không giải thích."
    )

    config = {
        "system_instruction": system_instruction,
        "temperature": 0.1,
        "thinking_config": {"thinking_level": "minimal"}
    }

    prompt = f" Chuan hoa doan review sau theo cac quy tac da cho: {text_input}"
    try:
        normalized_text = api_with_retry(prompt, config)
        return normalized_text
    except Exception as e:
        # neu co loi, in ra phan dau cua review va tra ve rong
        print(f"Da có loi xay ra khi xu ly API: {text_input[:50]}. Loi:{e}")
        return None

def normalize_with_cache(text_input):
    if text_input in CACHE:
        return CACHE[text_input]
    result = normalize_with_gemini(text_input)
    # chi cache khi result ko rong
    if result:
        CACHE[text_input] = result
        save_cache()
    return result

# ham loc lay review tieng viet
def detect_language(review: str) -> str:
    try:
        vn_review = str(review).strip()
        if not vn_review:
            return None
        lang = detect(vn_review)
        if lang == "vi":
            return vn_review
        return None
    except Exception as e:
        # khong the nhan dien ngon ngu, tra ve chuoi rong 
        return None

def attach_negation(text):
    neg_word = ["chưa_từng", "không", "chưa", "chẳng", "chả"]
    for neg in neg_word:
        # ghep cac tu phu dinh de khong bi tokenize
        remove_space = re.escape(neg)
        text = re.sub(rf"(?<!\w){remove_space}\s+([^\s.,!?;:]+)", rf"{neg}_\1", text)
    return text

def process_for_svm(review):
    review = ViTokenizer.tokenize(review)
    review = attach_negation(review) 
    review = re.sub(r"\b\d+\s*(năm|tháng|ngày)\b", "<TIME>", review)
    review = re.sub(r"\b\d+\b", "<NUM>", review) # chi thay the so bang <NUM> sau khi gemini xu ly
    review = re.sub(r"[.,!?;:]", "", review) # xoa dau cau
    review = re.sub(r"\s+", " ", review).strip()
    return review

def process_for_phobert(review):
    review = word_tokenize(review, format="text")
    return review

def process_for_qwen(review):
    return review

# ham chinh de xu ly toan bo review theo 3 model: loc review tieng viet, chuan hoa voi gemini
def normalize_review(review: str):
    if not isinstance(review, str) or not review.strip():
        return None
    vn_review = detect_language(review)
    if not vn_review:
        return None

    # chuyen thanh chu thuong
    review = vn_review.lower()

    # chuan hoa unicode cho tieng viet
    review = unicodedata.normalize("NFKC", review)
    
    # bo cac ky tu icon, giu dau cau
    review = re.sub(r"[^\w\s.,!?]", " ", review, flags=re.UNICODE)

    # chuan hoa voi gemini
    normalized_review = normalize_with_cache(review)
    if not normalized_review:
        return None
    normalized_review = re.sub(r"\s+", " ", normalized_review) .strip() 
    return normalized_review

if __name__ == "__main__": 
    pass 