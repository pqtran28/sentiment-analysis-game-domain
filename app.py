import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from peft import PeftModel

# =========================================================
# CONFIG
# =========================================================
REPO_ID    = "viegosequel/qwen2.5-1.5b-game-review-absa"
MAX_LEN    = 256
NUM_ASPECTS = 10
NUM_CLASSES = 3

LABEL_COLUMNS = [
    'graphics', 'matchmaking', 'store & microtransactions',
    'technical_issue', 'mechanics', 'developer_support', 'event',
    'community', 'hero_design', 'difficulty'
]

SENTIMENT_MAP = {
    0: ("Không nhắc tới",        "⚪", ""),
    1: ("Bình thường / Tích cực", "😊", "green"),
    2: ("Tiêu cực / Chê",        "😡", "red"),
}

# =========================================================
# MODEL CLASS
# =========================================================
class QwenMultiAspect(nn.Module):
    def __init__(self, backbone, hidden_size):
        super().__init__()
        self.backbone   = backbone
        self.dropout    = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, NUM_ASPECTS * NUM_CLASSES)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        h = out.hidden_states[-1]                        # [B, T, H]

        # Last-token pooling (phù hợp với decoder-only LM)
        seq_lengths = attention_mask.sum(dim=1) - 1
        p = h[torch.arange(h.size(0)), seq_lengths]     # [B, H]

        logits = self.classifier(self.dropout(p))        # [B, 10*3]
        return logits.view(-1, NUM_ASPECTS, NUM_CLASSES) # [B, 10, 3]

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource(show_spinner=False)
def load_model():
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Backbone — load CPU (Spaces thường không có GPU)
    # Nếu có GPU thì đổi device_map={'': 0} và thêm BitsAndBytesConfig
    backbone = AutoModel.from_pretrained(
        REPO_ID,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )

    # Classifier head
    classifier_state = torch.hub.load_state_dict_from_url(
        f"https://huggingface.co/{REPO_ID}/resolve/main/classifier_head.pt",
        map_location="cpu",
        weights_only=True,
    )

    model = QwenMultiAspect(backbone, backbone.config.hidden_size)
    model.classifier.load_state_dict(classifier_state)
    model.eval()
    return tokenizer, model

# =========================================================
# PREDICT
# =========================================================
def predict(text, tokenizer, model):
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc['input_ids'],
            attention_mask=enc['attention_mask'],
        )  # [1, 10, 3]
    preds = torch.argmax(logits[0], dim=-1).cpu().numpy()  # [10]
    return preds

# =========================================================
# UI
# =========================================================
st.set_page_config(
    page_title="Game Review ABSA",
    page_icon="🎮",
    layout="wide",
)

st.title("🎮 Game Review — Phân tích cảm xúc đa khía cạnh (ABSA)")
st.write(
    "Phân tích **10 khía cạnh** của bình luận game "
    "bằng mô hình **Qwen2.5-1.5B + QLoRA**."
)
st.markdown("---")

with st.spinner("🔄 Đang nạp mô hình từ Hugging Face Hub..."):
    try:
        tokenizer, model = load_model()
        st.success("✅ Mô hình đã sẵn sàng!", icon="✅")
    except Exception as e:
        st.error(f"❌ Lỗi khi nạp mô hình: {e}")
        st.stop()

# ── Input ──
user_input = st.text_area(
    "Nhập bình luận game cần phân tích:",
    placeholder="Ví dụ: Đồ họa đẹp nhưng lag quá, matchmaking tệ toàn match người mạnh hơn...",
    height=130,
)

col_btn, col_clear = st.columns([1, 5])
run = col_btn.button("🚀 Phân tích", use_container_width=True)

if run:
    if not user_input.strip():
        st.warning("⚠️ Vui lòng nhập bình luận trước khi phân tích.")
    else:
        with st.spinner("🤖 Đang phân tích..."):
            preds = predict(user_input.strip(), tokenizer, model)

        st.markdown("### 📊 Kết quả phân tích")

        # ── Chia 2 cột, 5 aspects mỗi cột ──
        col1, col2 = st.columns(2)
        mentioned = []

        for i, aspect in enumerate(LABEL_COLUMNS):
            label_id                = int(preds[i])
            label_text, icon, color = SENTIMENT_MAP[label_id]
            target_col              = col1 if i < 5 else col2

            with target_col:
                if label_id == 0:
                    st.markdown(
                        f"<div style='padding:6px 0; color:#888;'>"
                        f"{icon} <b>{aspect}</b>: {label_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    text_color = "#2e7d32" if label_id == 1 else "#c62828"
                    bg_color   = "#e8f5e9" if label_id == 1 else "#ffebee"
                    st.markdown(
                        f"<div style='padding:6px 8px; margin:4px 0; "
                        f"background:{bg_color}; border-radius:6px; color:{text_color};'>"
                        f"{icon} <b>{aspect.upper()}</b>: {label_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    mentioned.append((aspect, label_text, icon))

        # ── Summary ──
        st.markdown("---")
        n_mentioned = len(mentioned)
        n_negative  = sum(1 for _, lbl, _ in mentioned if "Tiêu cực" in lbl)
        n_positive  = n_mentioned - n_negative

        m1, m2, m3 = st.columns(3)
        m1.metric("Khía cạnh được nhắc tới", f"{n_mentioned} / {NUM_ASPECTS}")
        m2.metric("😊 Tích cực / Bình thường", n_positive)
        m3.metric("😡 Tiêu cực", n_negative)

        if n_mentioned == 0:
            st.info("ℹ️ Bình luận này không nhắc tới khía cạnh nào rõ ràng.")

# ── Footer ──
st.markdown("---")
st.caption(f"Model: `{REPO_ID}` · Aspects: {NUM_ASPECTS} · Classes: {NUM_CLASSES}")