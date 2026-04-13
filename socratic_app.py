"""
Socratic Tutor — Philosophy of Science
RAG-powered Oxbridge-style tutorial system
Built by Dr. Neal Doran | BIO 310, Bryan College
"""

import streamlit as st
import pickle
import os
import anthropic
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Socratic Tutor — Philosophy of Science",
    page_icon="🔭",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS — Oxford tutorial aesthetic ────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond:ital,wght@0,400;0,600;1,400&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');

    html, body, [class*="css"] {
        font-family: 'EB Garamond', Georgia, serif;
        background-color: #faf8f4;
        color: #1a1a1a;
    }
    .main { background-color: #faf8f4; }

    h1, h2, h3 {
        font-family: 'Libre Baskerville', Georgia, serif;
        color: #1a1a1a;
    }

    .tutor-header {
        text-align: center;
        padding: 8px 0 4px 0;
        border-bottom: 2px solid #8b0000;
        margin-bottom: 4px;
    }

    .tutor-title {
        font-family: 'Libre Baskerville', serif;
        font-size: 2.1rem;
        font-weight: 700;
        color: #1a1a1a;
        letter-spacing: 0.02em;
        margin: 0;
    }

    .tutor-subtitle {
        font-family: 'EB Garamond', serif;
        font-size: 1.05rem;
        color: #666;
        font-style: italic;
        margin: 4px 0 0 0;
    }

    .tutor-author {
        font-family: 'EB Garamond', serif;
        font-size: 0.9rem;
        color: #888;
        margin: 2px 0 0 0;
    }

    .galileo-caption {
        font-family: 'EB Garamond', serif;
        font-size: 0.82rem;
        color: #777;
        font-style: italic;
        text-align: center;
        margin-top: 4px;
        margin-bottom: 16px;
    }

    .response-box {
        background: #fffff8;
        border-left: 4px solid #8b0000;
        border-radius: 2px;
        padding: 20px 24px;
        margin: 16px 0;
        font-family: 'EB Garamond', serif;
        font-size: 1.18rem;
        line-height: 1.75;
        color: #1a1a1a;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    .exchange-student {
        background: #f0ede8;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 8px 0 2px 0;
        font-size: 1rem;
        font-style: italic;
        color: #333;
    }

    .exchange-tutor {
        background: #fffff8;
        border-left: 3px solid #8b0000;
        padding: 10px 14px;
        margin: 2px 0 12px 0;
        font-size: 1rem;
        line-height: 1.65;
        color: #1a1a1a;
    }

    .exchange-label {
        font-size: 0.75rem;
        font-family: 'Libre Baskerville', serif;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #8b0000;
        margin-bottom: 3px;
    }

    .example-claim {
        font-family: 'EB Garamond', serif;
        font-size: 0.95rem;
        font-style: italic;
        color: #555;
        cursor: pointer;
    }

    .rate-counter {
        font-size: 0.78rem;
        color: #aaa;
        font-family: 'EB Garamond', serif;
        text-align: right;
    }

    footer-custom {
        font-family: 'EB Garamond', serif;
        font-size: 0.8rem;
        color: #999;
        text-align: center;
        margin-top: 32px;
        font-style: italic;
    }

    /* Override Streamlit button */
    .stButton > button {
        background-color: #8b0000 !important;
        color: white !important;
        font-family: 'Libre Baskerville', serif !important;
        font-size: 1rem !important;
        border: none !important;
        padding: 10px 32px !important;
        border-radius: 2px !important;
        letter-spacing: 0.05em !important;
    }
    .stButton > button:hover {
        background-color: #6b0000 !important;
    }

    /* Text area */
    .stTextArea textarea {
        font-family: 'EB Garamond', serif !important;
        font-size: 1.05rem !important;
        border: 1px solid #ccc !important;
        border-radius: 2px !important;
        background: #fffff8 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'query_count' not in st.session_state:
    st.session_state.query_count = 0
if 'thesis_text' not in st.session_state:
    st.session_state.thesis_text = ""

MAX_QUERIES = 30

# ── Load models and data ──────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_embed_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_vector_store():
    """Load exported chunks into a simple numpy vector store."""
    pkl_path = os.path.join(BASE_DIR, "chroma_export.pkl")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # Convert embeddings to numpy array for fast similarity search
    embeddings = np.array(data['embeddings'], dtype=np.float32)
    return {
        'embeddings': embeddings,
        'documents': data['documents'],
        'metadatas': data['metadatas'],
    }, len(data['documents'])

def cosine_similarity_search(query_embedding, store, n_results=3):
    """Pure numpy cosine similarity — no ChromaDB needed."""
    q = np.array(query_embedding, dtype=np.float32)
    q = q / (np.linalg.norm(q) + 1e-10)
    embs = store['embeddings']
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
    normed = embs / norms
    scores = normed @ q
    top_idx = np.argsort(scores)[::-1][:n_results]
    return [
        {
            'document': store['documents'][i],
            'metadata': store['metadatas'][i],
            'score': float(scores[i])
        }
        for i in top_idx
    ]

embed_model = load_embed_model()
vector_store, chunk_count = load_vector_store()

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="tutor-header">
    <p class="tutor-title">Socratic Tutor</p>
    <p class="tutor-subtitle">Philosophy of Science — RAG-powered Oxbridge Tutorial</p>
    <p class="tutor-author">Dr. Neal Doran &nbsp;|&nbsp; BIO 310, Bryan College</p>
</div>
""", unsafe_allow_html=True)

# Galileo image
img_path = os.path.join(BASE_DIR, "galileo_copernican.png")
if os.path.exists(img_path):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(img_path, use_column_width=True)
    st.markdown(
        "<p class='galileo-caption'>Galileo's Copernican system, "
        "<em>Dialogue Concerning the Two Chief World Systems</em> (1632)</p>",
        unsafe_allow_html=True
    )

st.markdown("""
<p style='font-family: EB Garamond, serif; font-size: 1.05rem; 
   color: #444; line-height: 1.7; margin-bottom: 20px;'>
Present a thesis or claim about the course readings. Your tutor will challenge you 
to defend it — as at Oxford, no answer will be given. Only harder questions.
</p>
""", unsafe_allow_html=True)

# ── Example claims ────────────────────────────────────────────
st.markdown("**Try a claim:**")
examples = [
    "Kuhn argues that science progresses purely through logic and evidence",
    "Polanyi says that knowledge is personal",
    "Both Kuhn and Polanyi agree that scientific revolutions are inevitable",
    "Normal science is just puzzle solving",
]
ec1, ec2 = st.columns(2)
for i, ex in enumerate(examples):
    if (ec1 if i % 2 == 0 else ec2).button(f'"{ex}"', key=f"ex_{i}"):
        st.session_state.thesis_text = ex
        st.rerun()

# ── Input area ────────────────────────────────────────────────
thesis = st.text_area(
    "Your thesis:",
    value=st.session_state.thesis_text,
    height=100,
    placeholder="State a claim about Kuhn, Polanyi, or the nature of scientific knowledge...",
    label_visibility="visible"
)
st.session_state.thesis_text = thesis

# Rate counter
remaining = MAX_QUERIES - st.session_state.query_count
st.markdown(
    f"<p class='rate-counter'>Exchanges remaining: {remaining}/{MAX_QUERIES}</p>",
    unsafe_allow_html=True
)

# ── Socratic system prompt ────────────────────────────────────
SYSTEM_PROMPT = """You are a Socratic tutor for BIO 310: Philosophy of Science at Bryan College, 
combining the Socratic method with the Oxbridge tutorial tradition. You have access to passages 
retrieved from the student's assigned course readings (Kuhn's "The Structure of Scientific 
Revolutions" and Polanyi's "Personal Knowledge").

Your role: Challenge the student to defend their thesis as if presenting to a tutor at Oxford. 
Play devil's advocate using the retrieved passages. Expect the student to refine their argument 
with each exchange.

RULES:
1. NEVER give the student the answer directly
2. Ask ONE probing question at a time
3. If the student makes a claim, demand they support it with evidence from the readings
4. If the student is vague, press for precision — increase intellectual pressure each time they repeat a vague claim
5. Quote SHORT phrases (under 10 words) from the retrieved passages to direct attention
6. If the student is on the wrong track, redirect through a question, never correction
7. Always ground your questions in the retrieved passages — never introduce outside material
8. After 2-3 exchanges, ask the student to anticipate the strongest objection to their own position

RETRIEVED PASSAGES:
{retrieved_passages}

CONVERSATION HISTORY:
{history}

STUDENT'S CURRENT CLAIM:
{student_input}

Respond with ONE Socratic question only. Be intellectually rigorous. Do not be gentle."""


def ask_tutor(student_input):
    """Full RAG pipeline: retrieve → prompt → generate."""
    q_emb = embed_model.encode(student_input).tolist()
    results = cosine_similarity_search(q_emb, vector_store, n_results=3)

    passages = []
    passage_meta = []
    for i, r in enumerate(results):
        doc  = r['document']
        meta = r['metadata']
        score = r['score']
        passages.append(
            f"[Passage {i+1} — {meta['source']} | relevance: {score:.2f}]\n{doc[:400]}"
        )
        passage_meta.append({
            'source': meta['source'],
            'relevance': round(score, 2),
            'text': doc[:300]
        })

    retrieved_str = "\n\n".join(passages)

    history_str = ""
    for ex in st.session_state.history[-4:]:
        history_str += f"Student: {ex['student']}\nTutor: {ex['tutor']}\n\n"

    prompt = SYSTEM_PROMPT.format(
        retrieved_passages=retrieved_str,
        history=history_str if history_str else "This is the first exchange.",
        student_input=student_input
    )

    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        st.error("Add ANTHROPIC_API_KEY to Streamlit secrets.")
        st.stop()

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.content[0].text.strip(), passage_meta


# ── Challenge button ──────────────────────────────────────────
if st.button("⚔ Challenge Me", type="primary") and thesis.strip():
    if st.session_state.query_count >= MAX_QUERIES:
        st.warning(f"Session limit of {MAX_QUERIES} exchanges reached. Refresh to begin a new session.")
    else:
        with st.spinner("Your tutor considers your claim..."):
            tutor_response, passages = ask_tutor(thesis)

        st.session_state.history.append({
            'student': thesis,
            'tutor': tutor_response,
            'passages': passages
        })
        st.session_state.query_count += 1
        st.session_state.thesis_text = ""

# ── Display latest response ───────────────────────────────────
if st.session_state.history:
    latest = st.session_state.history[-1]

    st.markdown(
        f'<div class="response-box">{latest["tutor"]}</div>',
        unsafe_allow_html=True
    )

    with st.expander("📖 Retrieved source passages"):
        for p in latest['passages']:
            src = p['source'].replace('_', ' ')
            st.markdown(f"**{src}** — relevance: {p['relevance']:.2f}")
            st.markdown(f"*{p['text']}...*")
            st.divider()

# ── Conversation history ──────────────────────────────────────
if len(st.session_state.history) > 1:
    st.markdown("---")
    st.markdown("**Previous exchanges this session:**")
    for ex in reversed(st.session_state.history[:-1]):
        st.markdown(
            f'<div class="exchange-label">Student</div>'
            f'<div class="exchange-student">{ex["student"]}</div>'
            f'<div class="exchange-label">Tutor</div>'
            f'<div class="exchange-tutor">{ex["tutor"]}</div>',
            unsafe_allow_html=True
        )

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='font-family: EB Garamond, serif; font-size: 0.82rem; "
    "color: #999; text-align: center; font-style: italic;'>"
    "Powered by Retrieval-Augmented Generation &nbsp;|&nbsp; "
    "Sources: Kuhn, <em>The Structure of Scientific Revolutions</em>; "
    "Polanyi, <em>Personal Knowledge</em> &nbsp;|&nbsp; "
    f"Database: {chunk_count} indexed passages</p>",
    unsafe_allow_html=True
)
