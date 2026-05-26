import streamlit as st
import torch
import torch.nn as nn
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# =========================================================
# DANH SÁCH ASPECTS
# =========================================================
ASPECTS = [
    'graphics', 'matchmaking', 'monetization', 'technical_issue',
    'mechanics', 'developer_support', 'sound_music', 'tutorial',
    'story', 'quest', 'community', 'character', 'difficulty'
]

SENTIMENT_MAP = {
    0: "Không nhắc tới ⚪",
    1: "Có nhắc tới (Bình thường/Khen) 😊",
    2: "Có nhắc tới (Chê/Hate) 😡"
}

def my_preprocess_function(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

# =========================================================
# 2. TẢI MÔ HÌNH CHUẨN SEQUENCE CLASSIFICATION
# =========================================================
REPO_ID = "viegosequel/qwen2.5-1.5b-game-review-absa"

@st.cache_resource
def load_model_from_hf():
    # Tải tokenizer từ repo của bạn
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    
    # Tải mô hình Sequence Classification với đúng num_labels=39 như lúc bạn train
    # Thư viện sẽ tự gộp cấu hình cấu trúc lớp đầu ra cho bạn
    model = AutoModelForSequenceClassification.from_pretrained(
        REPO_ID,  # Gọi thẳng Repo PEFT/HuggingFace sẽ tự động load Base + LoRA + Classifier Head 39 nhãn
        num_labels=39,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True
    )
    
    model.eval()
    return tokenizer, model

# =========================================================
# 🎮 3. GIAO DIỆN WEB STREAMLIT
# =========================================================
st.set_page_config(page_title="Game Review ABSA", page_icon="🎮", layout="wide")

st.title("🎮 Game Review Aspect-Based Sentiment Analysis (ABSA)")
st.write("Ứng dụng phân tích 13 khía cạnh game dựa trên mô hình Qwen 2.5-1.5B Sequence Classification (39 Labels).")
st.markdown("---")

with st.spinner("🔄 Đang nạp mô hình Qwen từ Hugging Face Hub (Vui lòng đợi)..."):
    tokenizer, model = load_model_from_hf()

user_input = st.text_area("Nhập bài đánh giá game cần phân tích:", placeholder="Ví dụ: Đồ họa game này đẹp mê hồn nhưng hút máu quá đáng...", height=120)

if st.button("🚀 Phân tích đa khía cạnh"):
    if user_input.strip() == "":
        st.warning("Vui lòng nhập nội dung!")
    else:
        with st.spinner("🤖 AI đang tính toán toán học..."):
            # 1. Tiền xử lý
            cleaned_text = my_preprocess_function(user_input)
            st.markdown(f"**Văn bản sau tiền xử lý:** `{cleaned_text}`")
            st.markdown("### 📊 Kết quả phân tích chi tiết từ mô hình:")
            
            # 2. Tokenize dữ liệu đầu vào
            inputs = tokenizer(cleaned_text, return_tensors="pt", truncation=True, max_length=512).to("cpu")
            
            # 3. Dự đoán qua mô hình
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits  # Logits này có kích thước thẳng là [1, 39]
                
                # Biến đổi tensor [1, 39] thành ma trận [13, 3] tương ứng với (13 khía cạnh x 3 nhãn cảm xúc)
                # Đây chính là hàm tách ngược lại logic hàm loss lúc bạn train
                reshaped_logits = logits.view(-1, 13, 3) 
                
                # Lấy chỉ số có điểm cao nhất cho từng khía cạnh
                predictions = torch.argmax(reshaped_logits, dim=-1).squeeze(0).tolist()
            
            # 4. Hiển thị kết quả ra giao diện thành 2 cột cho đẹp
            col1, col2 = st.columns(2)
            
            for i, aspect in enumerate(ASPECTS):
                prediction = predictions[i]
                status_text = SENTIMENT_MAP[prediction]
                
                with col1 if i < 7 else col2:
                    # Làm nổi bật nếu khía cạnh đó có được nhắc tới (nhãn 1 hoặc 2)
                    if prediction in [1, 2]:
                        st.markdown(f"**🔹 {aspect.upper()}** : {status_text}")
                    else:
                        st.text(f"🔸 {aspect}: {status_text}")