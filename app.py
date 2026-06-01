import streamlit as st
import torch
import json
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_REPO = "viegosequel/qwen-game-review-absa-instruct_v1"

LABEL_COLUMNS = [
    "graphics", "matchmaking", "store & microtransactions",
    "technical_issue", "mechanics", "developer_support",
    "event", "community", "hero_design", "difficulty",
]

TARGET_NAMES = ["not_mentioned", "neutral_pos", "negative"]
STR2INT = {n: i for i, n in enumerate(TARGET_NAMES)}
INT2STR = {i: n for i, n in enumerate(TARGET_NAMES)}

ASPECT_LABELS_VI = {
    "graphics":                   "🎨 Đồ họa",
    "matchmaking":                "⚔️  Ghép trận",
    "store & microtransactions":  "🛒 Cửa hàng / MTX",
    "technical_issue":            "🔧 Lỗi kỹ thuật",
    "mechanics":                  "🎮 Cơ chế gameplay",
    "developer_support":          "🛠️  Hỗ trợ từ NPT",
    "event":                      "🎉 Sự kiện",
    "community":                  "👥 Cộng đồng",
    "hero_design":                "🦸 Thiết kế tướng",
    "difficulty":                 "💀 Độ khó",
}

SENTIMENT_CONFIG = {
    "not_mentioned": {"label": "Không đề cập", "color": "#64748b", "bg": "#1e293b", "icon": "—"},
    "neutral_pos":   {"label": "Tích cực / Trung lập", "color": "#22d3ee", "bg": "#0c4a6e", "icon": "✅"},
    "negative":      {"label": "Tiêu cực",             "color": "#f87171", "bg": "#450a0a", "icon": "❌"},
}

SYSTEM_PROMPT = (
    "Bạn là một hệ thống phân tích cảm xúc theo khía cạnh (ABSA) chuyên biệt. "
    "Nhiệm vụ của bạn là đọc bình luận về game và trả về KẾT QUẢ DUY NHẤT "
    "là một chuỗi JSON hợp lệ chứa đánh giá cho đúng 10 khía cạnh sau: "
    + ", ".join(LABEL_COLUMNS) + ". "
    "Mỗi giá trị BẮT BUỘC phải là một trong: "
    "'not_mentioned', 'neutral_pos', 'negative'. "
    "KHÔNG được thêm bất kỳ văn bản, giải thích hay markdown nào ngoài JSON."
)

EXAMPLE_REVIEWS = [
    "game đẹp lắm nhưng ghép trận tệ quá, toàn thua vì đồng đội rank thấp",
    "cửa hàng toàn skin đẹp nhưng giá quá cao, nghèo không mua được",
    "tướng mới thiết kế siêu đẹp nhưng bị lỗi lag liên tục không chịu được",
    "sự kiện hay lắm nhưng hỗ trợ từ nhà phát triển quá chậm, bug report mãi không fix",
    "đồ họa cải thiện nhiều, cộng đồng toxic giảm hẳn rồi, game ngày càng hay",
]

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Game Review ABSA",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');
:root {
    --bg-deep:    #050a14;
    --bg-panel:   #0d1424;
    --bg-card:    #111a2e;
    --border:     #1e3a5f;
    --accent:     #00d4ff;
    --accent2:    #7c3aed;
    --text-main:  #e2e8f0;
    --text-muted: #64748b;
    --negative:   #f87171;
    --positive:   #22d3ee;
    --neutral:    #64748b;
}
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-deep) !important;
    font-family: 'Inter', sans-serif;
    color: var(--text-main);
}
[data-testid="stHeader"] { background: transparent !important; }
/* Hero */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: linear-gradient(135deg, #00d4ff 0%, #7c3aed 50%, #f87171 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.1;
}
.hero-sub {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}
.hero-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.25);
    color: var(--accent);
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    letter-spacing: 0.15em;
    margin-top: 0.75rem;
}
/* Divider */
.neon-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--accent), var(--accent2), transparent);
    margin: 1rem 0 2rem;
    opacity: 0.5;
}
/* Input area */
.input-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
[data-testid="stTextArea"] textarea {
    background: var(--bg-panel) !important;
    border: 1px solid var(--border) !important;
    border-radius: 4px !important;
    color: var(--text-main) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    line-height: 1.6 !important;
    transition: border-color 0.2s;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 1px rgba(0,212,255,0.2) !important;
}
/* Buttons */
[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0ea5e9, #7c3aed) !important;
    border: none !important;
    border-radius: 3px !important;
    color: white !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
}
[data-testid="stButton"] > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
[data-testid="stButton"] > button:active {
    transform: translateY(0) !important;
}
/* Example pills */
.example-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
}
.example-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    background: rgba(124,58,237,0.1);
    border: 1px solid rgba(124,58,237,0.3);
    color: #a78bfa;
    padding: 0.25rem 0.6rem;
    border-radius: 2px;
    cursor: pointer;
    transition: background 0.15s;
}
/* Results grid */
.results-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 1rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
}
.aspect-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.85rem 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    transition: border-color 0.2s;
    margin-bottom: 0.5rem;
}
.aspect-card.sentiment-negative {
    border-left: 3px solid var(--negative);
    background: rgba(248,113,113,0.04);
}
.aspect-card.sentiment-positive {
    border-left: 3px solid var(--positive);
    background: rgba(34,211,238,0.04);
}
.aspect-card.sentiment-neutral {
    border-left: 3px solid var(--neutral);
    opacity: 0.65;
}
.aspect-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    color: var(--text-main);
}
.sentiment-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    padding: 0.2rem 0.6rem;
    border-radius: 2px;
    text-transform: uppercase;
}
.sentiment-badge.neg { background: rgba(248,113,113,0.15); color: #f87171; border: 1px solid rgba(248,113,113,0.3); }
.sentiment-badge.pos { background: rgba(34,211,238,0.12); color: #22d3ee; border: 1px solid rgba(34,211,238,0.3); }
.sentiment-badge.nm  { background: rgba(100,116,139,0.12); color: #64748b; border: 1px solid rgba(100,116,139,0.2); }
/* Summary bar */
.summary-bar {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin-top: 1.5rem;
    display: flex;
    gap: 2rem;
    flex-wrap: wrap;
    align-items: center;
}
.summary-stat {
    text-align: center;
}
.summary-num {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1;
}
.summary-lbl {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    color: var(--text-muted);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}
/* Loading */
.loading-box {
    text-align: center;
    padding: 2rem;
    font-family: 'IBM Plex Mono', monospace;
    color: var(--accent);
    font-size: 0.8rem;
    letter-spacing: 0.15em;
}
/* Info cards */
.info-card {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 1rem;
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.6;
}
.info-card h4 {
    font-family: 'Rajdhani', sans-serif;
    color: var(--accent);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 0 0 0.5rem;
    font-size: 0.85rem;
}
/* Streamlit overrides */
section[data-testid="stSidebar"] { background: var(--bg-panel) !important; }
div[data-testid="stSelectbox"] > div { background: var(--bg-panel) !important; border-color: var(--border) !important; }
div[data-testid="metric-container"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 4px !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MODEL LOADING (cached)
# ──────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    load_kwargs = dict(trust_remote_code=True, torch_dtype=torch.float16 if device == "cuda" else torch.float32)
    if device == "cuda":
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"

    model = AutoModelForCausalLM.from_pretrained(MODEL_REPO, **load_kwargs)
    model.eval()
    return model, tokenizer, device

# ──────────────────────────────────────────────
# INFERENCE
# ──────────────────────────────────────────────
def build_chat_messages(review: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Bình luận: {review}"},
    ]

def predict_absa(text: str, model, tokenizer, device: str) -> dict:
    messages = build_chat_messages(text)
    tokenized = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if isinstance(tokenized, dict) or hasattr(tokenized, "data"):
        input_ids = tokenized["input_ids"].to(device)
    else:
        input_ids = tokenized.to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse JSON
    try:
        clean = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()
        pred_dict = json.loads(clean)
    except (json.JSONDecodeError, ValueError):
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        pred_dict = json.loads(match.group()) if match else {}

    return {col: pred_dict.get(col, "not_mentioned") for col in LABEL_COLUMNS}

# ──────────────────────────────────────────────
# UI HELPERS
# ──────────────────────────────────────────────
def render_results(results: dict):
    n_neg  = sum(1 for v in results.values() if v == "negative")
    n_pos  = sum(1 for v in results.values() if v == "neutral_pos")
    n_nm   = sum(1 for v in results.values() if v == "not_mentioned")

    st.markdown('<div class="results-header">📊 KẾT QUẢ PHÂN TÍCH ABSA</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    mentioned = [(k, v) for k, v in results.items() if v != "not_mentioned"]
    not_mentioned = [(k, v) for k, v in results.items() if v == "not_mentioned"]

    with col1:
        if mentioned:
            st.markdown("**Các khía cạnh được đề cập:**")
            for aspect, sentiment in mentioned:
                cfg = SENTIMENT_CONFIG[sentiment]
                card_cls = "sentiment-negative" if sentiment == "negative" else "sentiment-positive"
                badge_cls = "neg" if sentiment == "negative" else "pos"
                badge_lbl = cfg["label"]
                icon = cfg["icon"]
                name_vi = ASPECT_LABELS_VI.get(aspect, aspect)
                st.markdown(f"""
                <div class="aspect-card {card_cls}">
                    <span class="aspect-name">{name_vi}</span>
                    <span class="sentiment-badge {badge_cls}">{icon} {badge_lbl}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Không có khía cạnh nào được đề cập.")

    with col2:
        if not_mentioned:
            st.markdown("**Không được đề cập:**")
            for aspect, _ in not_mentioned:
                name_vi = ASPECT_LABELS_VI.get(aspect, aspect)
                st.markdown(f"""
                <div class="aspect-card sentiment-neutral">
                    <span class="aspect-name">{name_vi}</span>
                    <span class="sentiment-badge nm">— Không đề cập</span>
                </div>
                """, unsafe_allow_html=True)

    # Summary bar
    summary_html = f"""
    <div class="summary-bar">
        <div class="summary-stat">
            <div class="summary-num" style="color:#f87171">{n_neg}</div>
            <div class="summary-lbl">Tiêu cực</div>
        </div>
        <div class="summary-stat">
            <div class="summary-num" style="color:#22d3ee">{n_pos}</div>
            <div class="summary-lbl">Tích cực</div>
        </div>
        <div class="summary-stat">
            <div class="summary-num" style="color:#64748b">{n_nm}</div>
            <div class="summary-lbl">Không đề cập</div>
        </div>
        <div class="summary-stat">
            <div class="summary-num" style="color:#a78bfa">{len(LABEL_COLUMNS)}</div>
            <div class="summary-lbl">Tổng khía cạnh</div>
        </div>
    </div>
    """
    st.markdown(summary_html, unsafe_allow_html=True)

    # Raw JSON expander
    with st.expander("🔍 Raw JSON output"):
        st.code(json.dumps(results, ensure_ascii=False, indent=2), language="json")

# ──────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────
def main():
    # Hero
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Game Review ABSA</div>
        <div class="hero-sub">Aspect-Based Sentiment Analysis · Qwen2.5-1.5B Instruct</div>
        <div class="hero-badge">viegosequel/qwen-game-review-absa-instruct_v1</div>
    </div>
    <div class="neon-divider"></div>
    """, unsafe_allow_html=True)

    # Load model
    with st.spinner("Đang tải model... (lần đầu có thể mất vài phút)"):
        try:
            model, tokenizer, device = load_model()
            device_label = "GPU ✅" if device == "cuda" else "CPU ⚠️"
            st.markdown(f"""
            <div style="text-align:center; margin-bottom:1.5rem;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#22d3ee; letter-spacing:0.15em;">
                    MODEL LOADED · {device_label}
                </span>
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Lỗi tải model: {e}")
            return

    # Layout
    left, right = st.columns([3, 2], gap="large")

    with left:
        st.markdown('<div class="input-label">📝 Nhập bình luận game</div>', unsafe_allow_html=True)

        # Example pills
        st.markdown("**Ví dụ nhanh:**")
        cols = st.columns(len(EXAMPLE_REVIEWS))
        selected_example = None
        for i, (col, ex) in enumerate(zip(cols, EXAMPLE_REVIEWS)):
            with col:
                if st.button(f"#{i+1}", key=f"ex_{i}", help=ex):
                    selected_example = ex

        # Text area
        default_text = selected_example if selected_example else st.session_state.get("review_text", "")
        review_text = st.text_area(
            label="review_input",
            value=default_text,
            placeholder="Nhập bình luận game tiếng Việt vào đây...\nVí dụ: game đẹp lắm nhưng ghép trận tệ quá, toàn thua vì đồng đội rank thấp",
            height=160,
            label_visibility="collapsed",
            key="review_text",
        )

        char_count = len(review_text.strip())
        st.markdown(f'<div style="font-family:IBM Plex Mono,monospace;font-size:0.65rem;color:#475569;text-align:right">{char_count} ký tự</div>', unsafe_allow_html=True)

        analyze_btn = st.button("⚡ PHÂN TÍCH NGAY", use_container_width=True)

    with right:
        st.markdown("""
        <div class="info-card">
            <h4>📌 Về model</h4>
            Model Qwen2.5-1.5B Instruct được fine-tune với QLoRA cho bài toán ABSA trên <strong style="color:#22d3ee">game reviews tiếng Việt</strong>.
            <br><br>
            <strong>Val macro F1:</strong> <span style="color:#22d3ee">0.6629</span><br>
            <strong>Test macro F1:</strong> <span style="color:#22d3ee">0.6375</span><br>
            <strong>Training:</strong> 4,152 samples · 5 epochs
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>🎯 10 Khía cạnh phân tích</h4>
        """ + "".join([
            f'<div style="display:flex;justify-content:space-between;padding:0.2rem 0;border-bottom:1px solid #1e293b">'
            f'<span>{v}</span></div>'
            for v in ASPECT_LABELS_VI.values()
        ]) + """
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
            <h4>🏷️ Nhãn phân loại</h4>
            <span style="color:#f87171">❌ Tiêu cực</span> — người dùng phàn nàn<br>
            <span style="color:#22d3ee">✅ Tích cực / Trung lập</span> — khen hoặc bình thường<br>
            <span style="color:#64748b">— Không đề cập</span> — không nhắc đến
        </div>
        """, unsafe_allow_html=True)

    # Analysis
    if analyze_btn:
        if not review_text.strip():
            st.warning("⚠️ Vui lòng nhập bình luận trước khi phân tích.")
        else:
            st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
            with st.spinner("🔄 Đang chạy inference..."):
                try:
                    results = predict_absa(review_text.strip(), model, tokenizer, device)
                    st.markdown(f"""
                    <div style="background:rgba(0,212,255,0.05); border:1px solid rgba(0,212,255,0.2); border-radius:4px;
                                padding:0.75rem 1rem; margin-bottom:1rem; font-family:IBM Plex Mono,monospace; font-size:0.8rem; color:#94a3b8;">
                        <strong style="color:#22d3ee">INPUT:</strong> {review_text[:120]}{"..." if len(review_text) > 120 else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    render_results(results)
                except Exception as e:
                    st.error(f"❌ Lỗi inference: {e}")

    # Footer
    st.markdown("""
    <div style="text-align:center; margin-top:3rem; padding-top:1rem; border-top:1px solid #1e293b;
                font-family:IBM Plex Mono,monospace; font-size:0.65rem; color:#334155; letter-spacing:0.1em;">
        QWEN2.5-1.5B INSTRUCT · GAME ABSA · VIEGOSEQUEL · 2026
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()