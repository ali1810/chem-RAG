"""
app.py — ChemRAG Streamlit Frontend
Chemistry RAG assistant — Dr. Mushtaq Ali, KIT
"""
import streamlit as st
import plotly.graph_objects as go
import api_client as api

st.set_page_config(
    page_title="ChemRAG — Chemical Literature Assistant",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero {
    background: linear-gradient(135deg, #0a1628 0%, #1a2e4a 50%, #0d4f3c 100%);
    padding: 28px 36px; border-radius: 16px; color: white; margin-bottom: 20px;
}
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; letter-spacing: -0.02em; }
.hero p  { opacity: 0.7; margin: 0; font-size: 0.9rem; }

.rag-step {
    display: inline-block; padding: 6px 14px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 3px;
}
.step-1 { background: rgba(99,179,237,0.15); color: #63b3ed; border: 1px solid #63b3ed; }
.step-2 { background: rgba(72,187,120,0.15); color: #48bb78; border: 1px solid #48bb78; }
.step-3 { background: rgba(237,137,54,0.15);  color: #ed8936; border: 1px solid #ed8936; }
.step-4 { background: rgba(159,122,234,0.15); color: #9f7aea; border: 1px solid #9f7aea; }

.answer-box {
    background: linear-gradient(135deg, rgba(13,79,60,0.08), rgba(10,22,40,0.08));
    border: 1px solid #2d6a4f; border-radius: 12px; padding: 20px;
    margin: 12px 0;
}
.source-card {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-left: 4px solid #2d6a4f; border-radius: 8px;
    padding: 14px; margin: 8px 0;
}
.source-title { font-weight: 600; color: #1a202c; font-size: 0.9rem; }
.source-score { font-size: 0.78rem; color: #718096; }
.source-excerpt { font-size: 0.82rem; color: #4a5568; margin-top: 6px;
                  font-family: 'JetBrains Mono', monospace; }

.doc-card {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 14px; margin: 6px 0;
}
.doc-title { font-weight: 600; font-size: 0.88rem; color: #2d3748; }
.doc-meta  { font-size: 0.75rem; color: #718096; margin-top: 4px; }

.metric-box {
    background: white; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 16px; text-align: center;
}
.metric-num { font-size: 1.8rem; font-weight: 700; color: #2d6a4f; }
.metric-lbl { font-size: 0.75rem; color: #718096; margin-top: 4px; }

.pipeline-box {
    background: #f0fff4; border: 1px solid #9ae6b4;
    border-radius: 10px; padding: 16px; font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
}

section[data-testid="stSidebar"] { background: #0a1628; }
section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
section[data-testid="stSidebar"] .stRadio label { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.4rem'>🧪</div>
        <div style='font-size:1.05rem;font-weight:700;color:white'>ChemRAG</div>
        <div style='font-size:0.72rem;opacity:0.5;color:#aaa'>Chemical Literature Assistant</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠 Home",
        "💬 Ask ChemRAG",
        "📥 Ingest Documents",
        "📚 Document Library",
        "🔬 How It Works",
    ], label_visibility="collapsed")

    st.divider()

    # Health status
    try:
        h = api.get_health()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Docs:** {h['document_count']}")
            st.markdown(f"**Chunks:** {h['chunk_count']}")
        with col_b:
            st.markdown("🟢 **API**" if h["status"] == "ok" else "🔴 **API**")
            st.markdown("🟢 **Groq**" if h["groq_key_set"] else "🔴 **Groq**")
        if not h["index_ready"]:
            st.warning("⚠️ No documents yet — go to Ingest")
    except Exception:
        st.markdown("🔴 **API offline**")
        st.caption("Start uvicorn backend")

    st.divider()
    top_k = st.slider("Retrieved chunks (k)", 1, 10, 5)


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🧪 ChemRAG — Chemical Literature RAG Assistant</h1>
        <p>Retrieval-Augmented Generation over chemistry papers and PubChem compound data · Dr. Mushtaq Ali · KIT</p>
    </div>""", unsafe_allow_html=True)

    # RAG pipeline steps
    st.markdown("""
    <div style='margin:16px 0 8px'>
        <span class="rag-step step-1">① Embed Query</span>
        <span style='color:#718096'>→</span>
        <span class="rag-step step-2">② FAISS Retrieval</span>
        <span style='color:#718096'>→</span>
        <span class="rag-step step-3">③ Context Assembly</span>
        <span style='color:#718096'>→</span>
        <span class="rag-step step-4">④ LLM Generation</span>
    </div>""", unsafe_allow_html=True)

    # Metrics
    try:
        h = api.get_health()
        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(f'<div class="metric-box"><div class="metric-num">{h["document_count"]}</div><div class="metric-lbl">Documents</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-box"><div class="metric-num">{h["chunk_count"]}</div><div class="metric-lbl">Indexed Chunks</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-box"><div class="metric-num">384</div><div class="metric-lbl">Embedding Dim</div></div>', unsafe_allow_html=True)
        c4.markdown(f'<div class="metric-box"><div class="metric-num">FAISS</div><div class="metric-lbl">Vector Store</div></div>', unsafe_allow_html=True)
    except Exception:
        st.error("API not running — start the FastAPI backend")

    st.write("")
    st.subheader("🚀 Quick Start")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
        **Step 1 — Ingest documents**
        Go to **📥 Ingest Documents** → click **Load Sample Docs**
        (5 chemistry documents loaded instantly)

        **Step 2 — Ask a question**
        Go to **💬 Ask ChemRAG** → type your question → get a grounded answer with citations
        """)
    with col_r:
        st.markdown("**Example questions:**")
        try:
            examples = api.get_examples()
            for ex in examples[:5]:
                if st.button(f"→ {ex}", key=f"home_{ex[:20]}", use_container_width=True):
                    st.session_state["prefill_question"] = ex
                    st.session_state["goto_query"] = True
                    st.rerun()
        except Exception:
            st.info("Start API to load examples")

    st.subheader("🏗️ Architecture")
    st.markdown("""
    <div class="pipeline-box">
User Query → [sentence-transformers: all-MiniLM-L6-v2]
           → Query Embedding (384-dim float32)
           → [FAISS IndexFlatIP] → Top-k chunks (cosine similarity)
           → Context Assembly (retrieved chunks + prompt template)
           → [Groq: llama-3.1-8b-instant] → Grounded Answer
           → Return: answer + cited sources + retrieval scores
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ASK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 Ask ChemRAG":
    st.header("💬 Ask ChemRAG")
    st.caption("Ask any chemistry question — answered from indexed documents with citations.")

    # Pre-fill from home
    default_q = st.session_state.pop("prefill_question", "")

    question = st.text_input(
        "Your chemistry question",
        value=default_q,
        placeholder="e.g. What is the relationship between LogP and drug solubility?"
    )

    col_ask, col_clear = st.columns([3, 1])
    with col_ask:
        ask_clicked = st.button("🔍 Ask ChemRAG", type="primary", use_container_width=True)
    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("last_result", None)
            st.rerun()

    # Example questions
    st.caption("Quick examples:")
    try:
        examples = api.get_examples()
        cols = st.columns(3)
        for i, ex in enumerate(examples[:6]):
            with cols[i % 3]:
                if st.button(ex[:45] + ("…" if len(ex) > 45 else ""),
                             key=f"ex_{i}", use_container_width=True):
                    st.session_state["prefill_question"] = ex
                    st.rerun()
    except Exception:
        pass

    # Run query
    if ask_clicked and question.strip():
        with st.spinner("🔍 Retrieving relevant documents... 💭 Generating answer..."):
            try:
                result = api.query(question.strip(), top_k=top_k)
                st.session_state["last_result"] = result
            except Exception as e:
                st.error(f"Query failed: {e}")
                st.stop()

    result = st.session_state.get("last_result")
    if not result:
        st.stop()

    st.divider()

    # Answer
    grounded = result.get("grounded", False)
    retrieval_count = result.get("retrieval_count", 0)

    st.markdown(f"""
    <div style='display:flex;gap:8px;align-items:center;margin-bottom:8px'>
        <span class="rag-step step-{'2' if grounded else '1'}">
            {'✅ Grounded answer' if grounded else '⚠️ No documents found'}
        </span>
        <span style='font-size:0.78rem;color:#718096'>
            {retrieval_count} chunks retrieved · Model: {result.get('model_used','—')}
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-box">
        <div style='font-size:0.82rem;color:#2d6a4f;font-weight:600;margin-bottom:8px'>
            🤖 ChemRAG Answer
        </div>
        <div style='font-size:0.95rem;line-height:1.7;color:#1a202c'>
            {result["answer"].replace(chr(10), "<br>")}
        </div>
    </div>""", unsafe_allow_html=True)

    # Sources
    sources = result.get("sources", [])
    if sources:
        st.subheader(f"📚 Sources ({len(sources)})")

        # Relevance chart
        if len(sources) > 1:
            fig = go.Figure(go.Bar(
                x=[s["score"] for s in sources],
                y=[s["title"][:40] + "…" if len(s["title"]) > 40 else s["title"]
                   for s in sources],
                orientation="h",
                marker_color=["#2d6a4f" if i == 0 else "#68d391" for i in range(len(sources))],
                text=[f"{s['score']:.3f}" for s in sources],
                textposition="outside",
            ))
            fig.update_layout(
                title="Retrieval Scores (cosine similarity)",
                height=max(200, len(sources) * 45),
                margin=dict(l=0, r=60, t=40, b=20),
                xaxis_title="Score",
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

        for s in sources:
            with st.expander(f"📄 {s['title']}  (score: {s['score']:.4f})", expanded=False):
                st.markdown(f"**Source:** {s['source']}")
                st.markdown(f"**Excerpt:**")
                st.markdown(f"""
                <div class="source-excerpt">{s['text_excerpt']}</div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📥 Ingest Documents":
    st.header("📥 Ingest Documents")
    st.caption("Add chemistry documents to the RAG knowledge base.")

    tab1, tab2, tab3 = st.tabs([
        "📦 Sample Documents", "🔬 PubChem CIDs", "📝 Paste Text"
    ])

    with tab1:
        st.markdown("""
        Load 5 built-in chemistry documents covering:
        - Aqueous solubility in drug discovery
        - Retrosynthesis and CASP
        - Chemical NER in scientific literature
        - ChemBERT and molecular transformers
        - RAG for scientific question answering
        """)
        if st.button("⚡ Load Sample Documents", type="primary"):
            with st.spinner("Ingesting sample documents..."):
                try:
                    r = api.ingest_samples()
                    st.success(f"✅ {r['message']}")
                    st.metric("Total chunks indexed", r["total_chunks"])
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    with tab2:
        st.markdown("Enter PubChem Compound IDs (CIDs) to ingest:")
        cid_input = st.text_area(
            "CIDs (comma or newline separated)",
            value="2244, 5090, 4091, 3672, 60961",
            help="2244=Aspirin, 5090=Caffeine, 4091=Paracetamol, 3672=Ibuprofen"
        )
        if st.button("🔬 Ingest from PubChem", type="primary"):
            try:
                raw = cid_input.replace("\n", ",").split(",")
                cids = [int(x.strip()) for x in raw if x.strip().isdigit()]
                if not cids:
                    st.error("No valid CIDs found")
                else:
                    with st.spinner(f"Fetching {len(cids)} compounds from PubChem..."):
                        r = api.ingest_pubchem(cids)
                        st.success(f"✅ {r['message']}")
            except Exception as e:
                st.error(f"Failed: {e}")

    with tab3:
        title = st.text_input("Document title", placeholder="e.g. Review of SMILES Notation")
        source = st.text_input("Source URL", value="manual", placeholder="https://...")
        text = st.text_area(
            "Document text",
            height=200,
            placeholder="Paste your chemistry paper abstract or text here..."
        )
        if st.button("📝 Ingest Text", type="primary"):
            if not title or not text:
                st.error("Title and text are required")
            else:
                with st.spinner("Ingesting..."):
                    try:
                        r = api.ingest_text(title, text, source)
                        st.success(f"✅ {r['message']}")
                    except Exception as e:
                        st.error(f"Failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📚 Document Library":
    st.header("📚 Document Library")
    st.caption("All documents currently in the RAG knowledge base.")

    try:
        docs = api.list_documents()
        if not docs:
            st.info("No documents ingested yet. Go to **📥 Ingest Documents** to add some.")
        else:
            st.metric("Total documents", len(docs))
            for doc in docs:
                with st.expander(f"📄 {doc['title']}  ({doc['chunk_count']} chunks)", expanded=False):
                    st.markdown(f"**Source:** {doc['source']}")
                    st.markdown(f"**Doc ID:** `{doc['doc_id']}`")
                    st.markdown(f"**Preview:** {doc['preview']}...")
    except Exception as e:
        st.error(f"Failed to load documents: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 How It Works":
    st.header("🔬 How ChemRAG Works")

    for title, content in {
        "① Document Ingestion Pipeline": """
**Input:** Chemistry papers, PubChem abstracts, raw text

**Process:**
1. Text is split into overlapping chunks (400 words, 50-word overlap)
2. Sentence-aware splitting avoids cutting mid-sentence
3. Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
4. Embeddings are L2-normalised so cosine similarity = dot product
5. Stored in FAISS IndexFlatIP + JSON metadata sidecar
6. Persisted to disk for reuse across sessions

**Why chunking with overlap?** Prevents context loss at chunk boundaries.
The 50-word overlap ensures that information spanning two chunks is captured by both.
""",
        "② FAISS Vector Retrieval": """
**Why FAISS?** Facebook AI Similarity Search — production-grade, in-process,
no external service. Used at Meta scale. Perfect for demos and production.

**Index type:** `IndexFlatIP` (inner product = cosine similarity on normalised vectors)

**Process:**
1. User query is embedded with the same sentence-transformer model
2. FAISS searches for top-k most similar chunk vectors
3. Returns chunk text + metadata + similarity score
4. Score range: 0.0 (unrelated) to 1.0 (identical)

**Alternative indices for production scale:**
- `IndexIVFFlat` — faster at >1M vectors (inverted file)
- `IndexHNSW` — approximate NN, very fast, slight accuracy trade-off
""",
        "③ Context Assembly & Prompt Engineering": """
**Retrieved chunks** are formatted into a structured context block:

```
[Document 1: Title]
...chunk text...
(Source: URL)

[Document 2: Title]
...chunk text...
```

**System prompt instructs the LLM to:**
- Answer ONLY from the provided context
- Flag when context is insufficient
- Always cite document titles
- Be precise and scientific

**Temperature = 0.1** — low temperature for factual, reproducible answers.
""",
        "④ LLM Generation (Groq)": """
**Model:** `llama-3.1-8b-instant` via Groq API

**Why Groq?**
- Free tier, very fast (tokens/sec much faster than OpenAI)
- Production-grade API — same pattern as OpenAI
- Easy to swap to GPT-4, Claude, or local Ollama

**Grounding mechanism:**
The system prompt + context-only instruction prevents hallucination.
If the answer isn't in the retrieved documents, the model says so.

**Output:** Natural language answer + which documents it used.
""",
        "⑤ Evaluation — How to measure RAG quality": """
**Retrieval metrics:**
- Recall@k — does the correct document appear in top-k?
- MRR (Mean Reciprocal Rank) — how high is the correct document ranked?
- NDCG — normalised discounted cumulative gain

**Generation metrics (RAGAS framework):**
- **Faithfulness** — does the answer stay within retrieved context?
- **Answer relevance** — does the answer address the question?
- **Context precision** — are retrieved chunks actually relevant?

**For Elsevier's enrichment pipelines:**
These same metrics apply — precision/recall on entity extraction,
human expert validation against gold-standard annotations.
""",
    }.items():
        with st.expander(title, expanded=False):
            st.markdown(content)

    st.subheader("🗂️ Tech Stack")
    cols = st.columns(4)
    for col, (name, detail) in zip(cols, [
        ("sentence-transformers", "all-MiniLM-L6-v2\n384-dim embeddings"),
        ("FAISS", "IndexFlatIP\nCosine similarity"),
        ("FastAPI", "REST API\nAsync Python"),
        ("Groq", "llama-3.1-8b\nFast LLM inference"),
    ]):
        col.markdown(f"""
        <div class="metric-box">
            <div style='font-weight:700;color:#2d3748;font-size:0.85rem'>{name}</div>
            <div style='font-size:0.72rem;color:#718096;margin-top:4px;white-space:pre-line'>{detail}</div>
        </div>""", unsafe_allow_html=True)
