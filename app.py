import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Internship Support Bot",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Neon theme CSS
# -----------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&display=swap');

:root {
    --neon-cyan: #00fff7;
    --neon-pink: #ff00e6;
    --neon-purple: #9d00ff;
    --bg-dark: #05050c;
    --bg-panel: #0d0d1a;
}

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
}

/* App background */
.stApp {
    background: radial-gradient(circle at 20% 20%, #12002b 0%, #05050c 45%, #000000 100%);
    color: #e8e8ff;
}

/* Title */
.neon-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    text-align: center;
    color: #fff;
    text-shadow:
        0 0 5px var(--neon-cyan),
        0 0 15px var(--neon-cyan),
        0 0 30px var(--neon-purple),
        0 0 60px var(--neon-purple);
    margin-bottom: 0;
    letter-spacing: 2px;
}

.neon-subtitle {
    text-align: center;
    color: var(--neon-pink);
    font-size: 1.05rem;
    text-shadow: 0 0 8px var(--neon-pink);
    margin-top: 4px;
    margin-bottom: 25px;
    letter-spacing: 1px;
}

/* Divider glow line */
.neon-divider {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-pink), var(--neon-purple), transparent);
    box-shadow: 0 0 15px var(--neon-cyan);
    margin-bottom: 30px;
    border: none;
}

/* Chat container card */
.chat-shell {
    background: rgba(13, 13, 26, 0.75);
    border: 1px solid rgba(0, 255, 247, 0.35);
    border-radius: 18px;
    padding: 20px 22px 8px 22px;
    box-shadow:
        0 0 20px rgba(0, 255, 247, 0.12),
        0 0 45px rgba(157, 0, 255, 0.10) inset;
    margin-bottom: 18px;
}

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background: rgba(20, 20, 40, 0.85);
    border-radius: 14px;
    border: 1px solid rgba(0, 255, 247, 0.25);
    box-shadow: 0 0 12px rgba(0, 255, 247, 0.10);
    padding: 4px 6px;
    margin-bottom: 10px;
}

/* User vs assistant accent (Streamlit assigns data-testid based on avatar order, so we style all consistently) */
[data-testid="stChatMessageContent"] p {
    color: #f0f0ff;
    font-size: 1.02rem;
}

/* Chat input box */
[data-testid="stChatInput"] textarea {
    background: rgba(10, 10, 20, 0.9) !important;
    border: 1px solid var(--neon-cyan) !important;
    box-shadow: 0 0 12px rgba(0, 255, 247, 0.35) !important;
    color: #fff !important;
    border-radius: 12px !important;
}

/* Metric-style badges row */
.badge-row {
    display: flex;
    justify-content: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 22px;
}
.neon-badge {
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid var(--neon-cyan);
    color: var(--neon-cyan);
    box-shadow: 0 0 10px rgba(0, 255, 247, 0.35);
    background: rgba(0, 255, 247, 0.06);
}
.neon-badge.pink {
    border-color: var(--neon-pink);
    color: var(--neon-pink);
    box-shadow: 0 0 10px rgba(255, 0, 230, 0.35);
    background: rgba(255, 0, 230, 0.06);
}
.neon-badge.purple {
    border-color: var(--neon-purple);
    color: #d9a8ff;
    box-shadow: 0 0 10px rgba(157, 0, 255, 0.35);
    background: rgba(157, 0, 255, 0.06);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a18 0%, #05050c 100%);
    border-right: 1px solid rgba(0, 255, 247, 0.25);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: var(--neon-cyan);
    text-shadow: 0 0 8px rgba(0, 255, 247, 0.6);
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] li {
    color: #cfcfe8;
}

/* Confidence pill under bot replies */
.conf-pill {
    display: inline-block;
    margin-top: 4px;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}
.conf-high { background: rgba(0,255,150,0.12); color: #00ffb0; border: 1px solid #00ffb0; }
.conf-low { background: rgba(255,80,80,0.12); color: #ff6b6b; border: 1px solid #ff6b6b; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-thumb { background: var(--neon-purple); border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load artifacts + models (cached so this runs only once)
# -----------------------------
@st.cache_resource(show_spinner="Booting up the support bot...")
def load_everything():
    with open("chatbot_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)

    embed_model = SentenceTransformer(artifacts["model_name"])

    return artifacts, embed_model


artifacts, embed_model = load_everything()

query_embeddings = artifacts["query_embeddings"]
all_queries_df = artifacts["all_queries_df"]
answer_lookup = artifacts["answer_lookup"]
SIMILARITY_THRESHOLD = artifacts["similarity_threshold"]


def get_bot_response(user_query):
    query_embedding = embed_model.encode([user_query])
    similarities = cosine_similarity(query_embedding, query_embeddings)[0]
    best_idx = np.argmax(similarities)
    best_score = float(similarities[best_idx])

    if best_score < SIMILARITY_THRESHOLD:
        return {
            "answer": "I'm not fully sure about that one. Please reach out to the support team for help with this query.",
            "confidence": best_score,
            "category": None,
        }

    matched_row = all_queries_df.iloc[best_idx]
    answer_id = matched_row["answer_id"]

    return {
        "answer": answer_lookup[answer_id],
        "confidence": best_score,
        "category": matched_row["category"],
    }


# -----------------------------
# Sidebar — info only
# -----------------------------
with st.sidebar:
    st.markdown("## 🛰️ About This Project")
    st.markdown(
        "An AI-powered support assistant that answers intern queries "
        "instantly using semantic search and natural language generation."
    )

    st.markdown("### ⚙️ Tech Stack")
    st.markdown(
        "- **Retrieval:** Sentence-BERT (`all-MiniLM-L6-v2`)\n"
        "- **Matching:** Cosine Similarity\n"
        "- **Interface:** Streamlit"
    )

    st.markdown("### 🗂️ Topics Covered")
    st.markdown(
        "Task submission • Deadlines • Mentor contact • Certificates • "
        "Attendance • Profile updates • Dashboard issues • Login & password • "
        "Internship duration • Task evaluation • Resubmission • Feedback • "
        "Completion status • Technical issues • Team projects • GitHub & "
        "Streamlit submission • Project requirements • AI usage policy"
    )

    st.markdown("### 🔎 How It Works")
    st.markdown(
        "1. Your message is converted into an embedding\n"
        "2. It's matched against a FAQ + support ticket dataset\n"
        "3. The closest, verified answer is retrieved and returned"
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()


# -----------------------------
# Main area — header
# -----------------------------
st.markdown('<div class="neon-title">💠 INTERNSHIP SUPPORT BOT</div>', unsafe_allow_html=True)
st.markdown('<div class="neon-subtitle">Real-time answers for intern queries, powered by NLP</div>', unsafe_allow_html=True)
st.markdown('<hr class="neon-divider">', unsafe_allow_html=True)

st.markdown("""
<div class="badge-row">
    <span class="neon-badge">🧠 Semantic Search</span>
    <span class="neon-badge pink">⚡ Real-Time</span>
    <span class="neon-badge purple">✅ Verified Answers</span>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Chat state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi! I'm your internship support bot 👋 Ask me anything about tasks, deadlines, certificates, or your dashboard.", "confidence": None}
    ]

# -----------------------------
# Render chat history
# -----------------------------
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("confidence") is not None:
            conf = msg["confidence"]
            css_class = "conf-high" if conf >= SIMILARITY_THRESHOLD else "conf-low"
            st.markdown(
                f'<span class="conf-pill {css_class}">confidence: {conf:.2f}</span>',
                unsafe_allow_html=True,
            )

st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Chat input
# -----------------------------
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input, "confidence": None})

    with st.spinner("Thinking..."):
        result = get_bot_response(user_input)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "confidence": result["confidence"],
    })

    st.rerun()