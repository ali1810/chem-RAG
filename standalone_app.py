"""
standalone_app.py — ChemRAG Standalone for Streamlit Cloud
Runs entirely in Streamlit — no FastAPI backend needed.
Embedding + FAISS + Groq all run directly in the Streamlit process.
"""
import os
import re
import uuid
import json
import requests
import streamlit as st
import numpy as np
import plotly.graph_objects as go

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
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; }
.hero p  { opacity: 0.7; margin: 0; font-size: 0.9rem; }
.rag-step { display:inline-block; padding:6px 14px; border-radius:20px; font-size:0.78rem; font-weight:600; margin:3px; }
.step-1 { background:rgba(99,179,237,0.15); color:#63b3ed; border:1px solid #63b3ed; }
.step-2 { background:rgba(72,187,120,0.15); color:#48bb78; border:1px solid #48bb78; }
.step-3 { background:rgba(237,137,54,0.15);  color:#ed8936; border:1px solid #ed8936; }
.step-4 { background:rgba(159,122,234,0.15); color:#9f7aea; border:1px solid #9f7aea; }
.answer-box {
    background:linear-gradient(135deg,rgba(13,79,60,0.08),rgba(10,22,40,0.08));
    border:1px solid #2d6a4f; border-radius:12px; padding:20px; margin:12px 0;
}
.source-excerpt { font-size:0.82rem; color:#4a5568; margin-top:6px; font-family:'JetBrains Mono',monospace; }
.metric-box { background:white; border:1px solid #e2e8f0; border-radius:10px; padding:16px; text-align:center; }
.metric-num { font-size:1.8rem; font-weight:700; color:#2d6a4f; }
.metric-lbl { font-size:0.75rem; color:#718096; margin-top:4px; }
section[data-testid="stSidebar"] { background:#0a1628; }
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 400
CHUNK_OVERLAP   = 50
TOP_K           = 5
GROQ_MODEL      = "llama-3.1-8b-instant"

def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


# ── Embedding ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)

def embed(texts):
    model = load_embedder()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vecs, dtype=np.float32)


# ── FAISS store (session state) ───────────────────────────────────────────────
def get_store():
    if "faiss_index" not in st.session_state:
        import faiss
        dim = 384
        st.session_state["faiss_index"]  = faiss.IndexFlatIP(dim)
        st.session_state["faiss_chunks"] = []
    return st.session_state["faiss_index"], st.session_state["faiss_chunks"]


def add_to_store(chunks_data, embeddings):
    index, chunks = get_store()
    index.add(embeddings)
    chunks.extend(chunks_data)


def search_store(query_vec, top_k=5):
    index, chunks = get_store()
    if index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    scores, indices = index.search(query_vec.reshape(1,-1), k)
    return [(chunks[i], float(scores[0][j]))
            for j, i in enumerate(indices[0]) if i >= 0 and i < len(chunks)]


def store_count():
    index, _ = get_store()
    return index.ntotal


def get_documents():
    _, chunks = get_store()
    docs = {}
    for c in chunks:
        did = c["doc_id"]
        if did not in docs:
            docs[did] = {"doc_id": did, "title": c["title"],
                         "source": c["source"], "chunk_count": 0,
                         "preview": c["text"][:200]}
        docs[did]["chunk_count"] += 1
    return list(docs.values())


def ingested_doc_ids():
    _, chunks = get_store()
    return {c["doc_id"] for c in chunks}


# ── Chunking ──────────────────────────────────────────────────────────────────
def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip()) if s.strip()]

def chunk_text(text):
    sentences = split_sentences(text)
    chunks, current = [], []
    size = 0
    for sent in sentences:
        words = sent.split()
        if size + len(words) > CHUNK_SIZE and current:
            chunks.append(" ".join(current))
            current = current[-CHUNK_OVERLAP:] + words
            size = len(current)
        else:
            current.extend(words)
            size += len(words)
    if current:
        chunks.append(" ".join(current))
    return chunks

def ingest_document(doc_id, title, text, source, metadata=None):
    if doc_id in ingested_doc_ids():
        return 0
    raw_chunks = chunk_text(text)
    if not raw_chunks:
        return 0
    chunks_data = [
        {"chunk_id": str(uuid.uuid4()), "doc_id": doc_id, "title": title,
         "source": source, "text": chunk, "chunk_index": i,
         "metadata": metadata or {}}
        for i, chunk in enumerate(raw_chunks)
    ]
    embeddings = embed([c["text"] for c in chunks_data])
    add_to_store(chunks_data, embeddings)
    return len(chunks_data)


# ── PubChem fetch ─────────────────────────────────────────────────────────────
def fetch_pubchem(cid):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        info = r.json().get("InformationList", {}).get("Information", [])
        title, texts = f"PubChem CID {cid}", []
        for i in info:
            if "Title" in i: title = i["Title"]
            if "Description" in i: texts.append(i["Description"])
        if not texts: return None
        # Properties
        props = ["MolecularFormula","MolecularWeight","IUPACName","CanonicalSMILES","XLogP"]
        pr = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{','.join(props)}/JSON",
            timeout=10)
        prop_text = ""
        if pr.ok:
            pd = pr.json().get("PropertyTable", {}).get("Properties", [{}])[0]
            parts = []
            if "IUPACName"      in pd: parts.append(f"IUPAC: {pd['IUPACName']}.")
            if "MolecularFormula" in pd: parts.append(f"Formula: {pd['MolecularFormula']}.")
            if "MolecularWeight" in pd: parts.append(f"MW: {pd['MolecularWeight']} g/mol.")
            if "CanonicalSMILES" in pd: parts.append(f"SMILES: {pd['CanonicalSMILES']}.")
            if "XLogP"          in pd: parts.append(f"LogP: {pd['XLogP']}.")
            prop_text = " ".join(parts) + " "
        return {"doc_id": f"pubchem_{cid}", "title": title,
                "text": prop_text + " ".join(texts),
                "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"}
    except Exception as e:
        return None


# ── Sample documents ──────────────────────────────────────────────────────────
SAMPLES = [
    ("Aqueous Solubility in Drug Discovery", """
Aqueous solubility is one of the most critical physicochemical properties in drug discovery.
Poor solubility is a major cause of drug attrition, with estimates suggesting that 40% of new
chemical entities suffer from low aqueous solubility. Aqueous solubility is defined as the maximum
amount of a substance that dissolves in a given volume of water at a specific temperature and pH.
It is typically expressed in mg/mL or as the log of molar solubility (LogS). Factors affecting
aqueous solubility include crystal packing energy, melting point, lipophilicity (LogP), hydrogen
bonding capacity, molecular weight, and polar surface area (PSA). Lipinski's rule of five states
that poor absorption is more likely when molecular weight exceeds 500 Da, LogP exceeds 5, hydrogen
bond donors exceed 5, and acceptors exceed 10. Machine learning models using ChemBERT and MPNN
architectures trained on PubChem and ChEMBL data achieve state-of-the-art solubility prediction.
Temperature and pH significantly affect solubility of ionisable compounds through the
Henderson-Hasselbalch equation. The ESOL model by Delaney uses linear regression on molecular
descriptors including LogP, molecular weight, and rotatable bonds.
    """.strip()),
    ("Retrosynthesis and Computer-Aided Synthesis Planning", """
Retrosynthesis is a problem-solving technique introduced by E.J. Corey for planning the synthesis
of complex molecules by working backwards from the target to simpler precursors. Computer-aided
synthesis planning (CASP) has been revolutionised by machine learning. Template-free methods use
sequence-to-sequence neural networks treating SMILES as sequences, similar to machine translation.
The USPTO-50K dataset, containing 50,000 patent reactions across 10 reaction classes, is the
standard benchmark. State-of-the-art models achieve over 85% Top-1 accuracy. Reaction classes
include heteroatom alkylation, acylation, C-C bond formation, heterocycle formation, protections,
deprotections, reductions, oxidations, functional group interconversion (FGI), and functional
group addition (FGA). Beam search generates multiple candidate reactant predictions ranked by
log-probability. SMILES augmentation generates multiple valid SMILES representations of the same
molecule and is a key technique for improving model generalisation. Reaxys Predictive
Retrosynthesis integrates AI-powered retrosynthesis with the world's largest curated reaction
database containing over 500 million reactions.
    """.strip()),
    ("Chemical Named Entity Recognition", """
Chemical named entity recognition (NER) identifies and classifies chemical entities in scientific
text including compound names, molecular formulas, SMILES strings, protein names, and disease
terms. Chemical NER is challenging due to diverse nomenclature — a compound may be referred to
by IUPAC name, common name, abbreviation, CAS number, or PubChem CID. Domain-specific transformer
models such as ChemBERT, MatBERT, and BioBERT significantly outperform general-purpose BERT.
Common annotation schemes include BIO (Beginning, Inside, Outside) and BIOES tagging. Evaluation
uses entity-level precision, recall, and F1. Key datasets include BC5CDR, CHEMDNER, and NLMChem.
Information extraction pipelines for Reaxys and Embase use NER as the first step, followed by
relation extraction identifying reactions between entities, entity linking normalising to database
identifiers, and attribute extraction for yield, temperature, and solvent. Hugging Face Transformers
provides the standard framework for fine-tuning chemical NER models on annotated corpora.
    """.strip()),
    ("Transformer Models for Chemistry — ChemBERT", """
Transformer models have transformed NLP and are increasingly applied to chemistry. BERT uses a
self-attention mechanism to encode contextual representations. For chemistry, SMILES strings are
tokenised where each atom, bond, and bracket is a token. ChemBERT is pre-trained on 77 million
SMILES strings from PubChem and learns molecular representations encoding structure and bonding.
Fine-tuned ChemBERT achieves strong performance on solubility, toxicity, and bioactivity prediction.
The Molecular Transformer by Schwaller et al. applies seq2seq transformers to reaction prediction,
treating reactant and product SMILES as source and target sequences with state-of-the-art benchmark
performance. Graph Neural Networks (GNNs) and Message Passing Neural Networks (MPNNs) complement
sequence-based models by representing molecules as graphs where atoms are nodes and bonds are edges.
Multi-modal architectures combining SMILES transformers with GNNs and molecular fingerprints achieve
the best performance by capturing different aspects of molecular structure. Hugging Face provides
pre-trained ChemBERT models for downstream fine-tuning.
    """.strip()),
    ("Retrieval-Augmented Generation for Scientific QA", """
Retrieval-Augmented Generation (RAG) combines information retrieval with large language model
generation to produce grounded, citation-backed answers. A user query retrieves relevant documents
from a knowledge base using dense vector similarity search, then a language model generates an
answer grounded in the retrieved evidence. RAG addresses hallucination by anchoring generation to
retrieved documents. The pipeline has four components. Document ingestion: documents are chunked
into 200-500 token passages with overlap, embedded using a sentence encoder. Indexing: embeddings
stored in FAISS, Pinecone, or OpenSearch for nearest neighbour retrieval. Retrieval: query is
encoded and top-k most similar chunks retrieved by cosine similarity. Generation: retrieved chunks
concatenated as context with the query for the LLM. Evaluation uses context relevance, faithfulness,
and answer relevance metrics. The RAGAS framework provides automated evaluation. For chemistry RAG,
domain-specific models like ChemBERT or SciBERT improve retrieval quality over general-purpose
sentence encoders.
    """.strip()),
]

def ingest_samples():
    total = 0
    for title, text in SAMPLES:
        doc_id = "sample_" + title.lower().replace(" ", "_")[:40]
        n = ingest_document(doc_id, title, text, "ChemRAG Demo")
        total += n
    return len(SAMPLES), total


# ── RAG pipeline ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ChemRAG, an expert chemistry research assistant.
Answer questions about chemistry, drug discovery, molecular properties, and reactions.
RULES: 1) Answer ONLY from provided context. 2) If context is insufficient, say so.
3) Cite document titles. 4) Do not invent chemical facts or data. 5) Be precise."""

def run_rag(question, top_k=5):
    query_vec = embed([question])[0]
    retrieved = search_store(query_vec, top_k=top_k)
    if not retrieved:
        return {
            "answer": "No documents indexed yet. Go to **📥 Ingest** and load sample documents first.",
            "sources": [], "grounded": False, "retrieval_count": 0
        }
    context = "\n\n".join(
        f"[Document {i+1}: {c['title']}]\n{c['text']}\n(Source: {c['source']})"
        for i, (c, _) in enumerate(retrieved)
    )
    prompt = f"Context:\n{context}\n\n---\nQuestion: {question}\n\nAnswer from context only, cite document titles."

    groq_key = get_groq_key()
    if not groq_key:
        answer = ("⚠️ No GROQ_API_KEY set. Add it in Streamlit Secrets.\n\n"
                  "**Retrieved context:**\n\n" +
                  "\n\n".join(f"**{c['title']}:** {c['text'][:300]}..."
                              for c, _ in retrieved[:3]))
    else:
        try:
            from groq import Groq
            client = Groq(api_key=groq_key)
            resp = client.chat.completions.create(
                model=GROQ_MODEL, temperature=0.1, max_tokens=600,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ]
            )
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            answer = f"LLM error: {e}"

    sources = []
    seen = set()
    for chunk, score in retrieved:
        if chunk["doc_id"] not in seen:
            seen.add(chunk["doc_id"])
            sources.append({"title": chunk["title"], "source": chunk["source"],
                            "excerpt": chunk["text"][:300] + "...", "score": round(score, 4)})
    return {"answer": answer, "sources": sources, "grounded": True,
            "retrieval_count": len(retrieved)}


# ── Example questions ─────────────────────────────────────────────────────────
EXAMPLES = [
    "What is the relationship between LogP and drug solubility?",
    "How does retrosynthesis work and what is USPTO-50K?",
    "Explain how ChemBERT represents chemical structures",
    "What is FAISS and how does it do vector similarity search?",
    "How does RAG prevent hallucination in LLMs?",
    "What reaction classes are in the USPTO-50K dataset?",
    "How does SMILES augmentation improve model performance?",
    "What is the role of beam search in retrosynthesis prediction?",
    "How do GNNs represent molecules differently from SMILES?",
    "What NLP challenges are specific to chemistry text?",
]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.4rem'>🧪</div>
        <div style='font-size:1.05rem;font-weight:700'>ChemRAG</div>
        <div style='font-size:0.72rem;opacity:0.5'>Chemical Literature Assistant</div>
        <div style='font-size:0.68rem;opacity:0.4;margin-top:4px'>Dr. Mushtaq Ali · KIT</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠 Home", "💬 Ask ChemRAG", "📥 Ingest", "📚 Library", "🔬 How It Works"
    ], label_visibility="collapsed")

    st.divider()
    chunk_count = store_count()
    doc_count   = len(get_documents())
    st.markdown(f"**📄 Docs:** {doc_count}  |  **🧩 Chunks:** {chunk_count}")
    groq_ok = bool(get_groq_key())
    st.markdown("🟢 **Groq ready**" if groq_ok else "🔴 **No Groq key**")
    if chunk_count == 0:
        st.warning("⚠️ No docs — go to Ingest")
    st.divider()
    top_k = st.slider("Retrieve k chunks", 1, 10, 5)


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🧪 ChemRAG — Chemical Literature RAG Assistant</h1>
        <p>Retrieval-Augmented Generation over chemistry papers and PubChem compound data · Dr. Mushtaq Ali · KIT</p>
    </div>""", unsafe_allow_html=True)

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

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in [
        (c1, doc_count, "Documents"),
        (c2, chunk_count, "Indexed Chunks"),
        (c3, "384", "Embedding Dim"),
        (c4, "FAISS", "Vector Store"),
    ]:
        col.markdown(f'<div class="metric-box"><div class="metric-num">{num}</div><div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.write("")
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Go to **📥 Ingest** → click **Load Sample Docs**
        2. Go to **💬 Ask ChemRAG** → ask your question
        3. See the grounded answer with citations
        """)
    with col_r:
        st.subheader("Example questions")
        for ex in EXAMPLES[:5]:
            if st.button(f"→ {ex[:55]}", key=f"h_{ex[:15]}", use_container_width=True):
                st.session_state["pq"] = ex
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ASK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 Ask ChemRAG":
    st.header("💬 Ask ChemRAG")
    default_q = st.session_state.pop("pq", "")
    question = st.text_input("Your chemistry question", value=default_q,
        placeholder="e.g. What is the relationship between LogP and drug solubility?")

    col_a, col_b = st.columns([3,1])
    with col_a:
        ask = st.button("🔍 Ask ChemRAG", type="primary", use_container_width=True)
    with col_b:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("result", None)
            st.rerun()

    st.caption("Quick examples:")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES[:6]):
        with cols[i%3]:
            if st.button(ex[:42]+"…" if len(ex)>42 else ex, key=f"e_{i}", use_container_width=True):
                st.session_state["pq"] = ex
                st.rerun()

    if ask and question.strip():
        with st.spinner("🔍 Retrieving... 💭 Generating..."):
            try:
                st.session_state["result"] = run_rag(question.strip(), top_k=top_k)
            except Exception as e:
                st.error(f"Error: {e}")

    result = st.session_state.get("result")
    if not result:
        st.stop()

    st.divider()
    st.markdown(f"""
    <div style='display:flex;gap:8px;align-items:center;margin-bottom:8px'>
        <span class="rag-step step-{'2' if result['grounded'] else '1'}">
            {'✅ Grounded answer' if result['grounded'] else '⚠️ No docs found'}
        </span>
        <span style='font-size:0.78rem;color:#718096'>
            {result['retrieval_count']} chunks retrieved · {GROQ_MODEL}
        </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="answer-box">
        <div style='font-size:0.82rem;color:#2d6a4f;font-weight:600;margin-bottom:8px'>🤖 ChemRAG Answer</div>
        <div style='font-size:0.95rem;line-height:1.7;color:#1a202c'>
            {result["answer"].replace(chr(10),"<br>")}
        </div>
    </div>""", unsafe_allow_html=True)

    sources = result.get("sources", [])
    if sources:
        st.subheader(f"📚 Sources ({len(sources)})")
        if len(sources) > 1:
            fig = go.Figure(go.Bar(
                x=[s["score"] for s in sources],
                y=[s["title"][:40]+"…" if len(s["title"])>40 else s["title"] for s in sources],
                orientation="h",
                marker_color=["#2d6a4f" if i==0 else "#68d391" for i in range(len(sources))],
                text=[f"{s['score']:.3f}" for s in sources], textposition="outside",
            ))
            fig.update_layout(title="Retrieval Scores (cosine similarity)",
                height=max(180, len(sources)*45),
                margin=dict(l=0,r=60,t=40,b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        for s in sources:
            with st.expander(f"📄 {s['title']}  (score: {s['score']:.4f})", expanded=False):
                st.markdown(f"**Source:** {s['source']}")
                st.markdown(f'<div class="source-excerpt">{s["excerpt"]}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📥 Ingest":
    st.header("📥 Ingest Documents")
    tab1, tab2, tab3 = st.tabs(["📦 Sample Docs", "🔬 PubChem CIDs", "📝 Paste Text"])

    with tab1:
        st.markdown("Load 5 built-in chemistry documents: solubility, retrosynthesis, NER, ChemBERT, RAG.")
        if st.button("⚡ Load Sample Documents", type="primary"):
            with st.spinner("Ingesting..."):
                n_docs, n_chunks = ingest_samples()
                st.success(f"✅ Ingested {n_docs} documents ({n_chunks} chunks)")
                st.rerun()

    with tab2:
        cid_input = st.text_area("PubChem CIDs (comma separated)",
            value="2244, 5090, 4091, 3672",
            help="2244=Aspirin, 5090=Caffeine, 4091=Paracetamol, 3672=Ibuprofen")
        if st.button("🔬 Ingest from PubChem", type="primary"):
            raw = cid_input.replace("\n",",").split(",")
            cids = [int(x.strip()) for x in raw if x.strip().isdigit()]
            if not cids:
                st.error("No valid CIDs")
            else:
                total = 0
                with st.spinner(f"Fetching {len(cids)} compounds..."):
                    for cid in cids:
                        doc = fetch_pubchem(cid)
                        if doc:
                            n = ingest_document(doc["doc_id"], doc["title"],
                                                doc["text"], doc["source"])
                            total += n
                st.success(f"✅ Ingested {len(cids)} compounds ({total} chunks)")
                st.rerun()

    with tab3:
        title  = st.text_input("Title", placeholder="e.g. Review of SMILES Notation")
        source = st.text_input("Source", value="manual")
        text   = st.text_area("Text", height=200,
            placeholder="Paste chemistry paper abstract or text here...")
        if st.button("📝 Ingest Text", type="primary"):
            if not title or not text:
                st.error("Title and text required")
            else:
                doc_id = f"manual_{uuid.uuid4().hex[:8]}"
                n = ingest_document(doc_id, title, text, source)
                st.success(f"✅ Ingested '{title}' ({n} chunks)")
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📚 Library":
    st.header("📚 Document Library")
    docs = get_documents()
    if not docs:
        st.info("No documents yet — go to **📥 Ingest**")
    else:
        st.metric("Total documents", len(docs))
        for doc in docs:
            with st.expander(f"📄 {doc['title']}  ({doc['chunk_count']} chunks)"):
                st.markdown(f"**Source:** {doc['source']}")
                st.markdown(f"**Preview:** {doc['preview']}...")


# ─────────────────────────────────────────────────────────────────────────────
# HOW IT WORKS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔬 How It Works":
    st.header("🔬 How ChemRAG Works")

    for title, body in [
        ("① Document Ingestion", """
Text is split into overlapping chunks (400 words, 50-word overlap) to avoid losing context at boundaries.
Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, L2-normalised).
Embeddings are stored in a FAISS IndexFlatIP — inner product on normalised vectors equals cosine similarity.
        """),
        ("② FAISS Vector Retrieval", """
The user query is embedded with the same model. FAISS searches for top-k most similar chunk vectors.
`IndexFlatIP` performs exact search — cosine similarity 0.0 (unrelated) to 1.0 (identical).
For production scale: `IndexIVFFlat` (inverted file, faster at >1M vectors) or `IndexHNSW` (approximate, very fast).
        """),
        ("③ Context Assembly & Prompt Engineering", """
Retrieved chunks formatted into a structured context block with document titles and source URLs.
System prompt instructs LLM to: answer only from context, cite titles, flag insufficient information.
Temperature = 0.1 for factual, reproducible chemistry answers.
        """),
        ("④ LLM Generation — Groq", """
Model: `llama-3.1-8b-instant` via Groq API (free, fast, OpenAI-compatible).
Grounding: context-only instruction prevents hallucination.
Swap: same code works with OpenAI GPT-4, Anthropic Claude, or local Ollama.
        """),
        ("⑤ RAG Evaluation Metrics", """
**Retrieval:** Recall@k, MRR, NDCG — does the correct document appear in top-k?
**Generation (RAGAS):** faithfulness (answer stays in context?), answer relevance, context precision.
**For Elsevier enrichment pipelines:** precision/recall on entity extraction, human expert validation against gold standard.
        """),
    ]:
        with st.expander(title):
            st.markdown(body)

    st.subheader("Tech Stack")
    cols = st.columns(4)
    for col, (name, detail) in zip(cols, [
        ("sentence-transformers", "all-MiniLM-L6-v2\n384-dim embeddings"),
        ("FAISS", "IndexFlatIP\nCosine similarity"),
        ("Groq API", "llama-3.1-8b-instant\nFast LLM inference"),
        ("Streamlit", "Interactive UI\nDirect deployment"),
    ]):
        col.markdown(f"""
        <div class="metric-box">
            <div style='font-weight:700;color:#2d3748;font-size:0.85rem'>{name}</div>
            <div style='font-size:0.72rem;color:#718096;margin-top:4px;white-space:pre-line'>{detail}</div>
        </div>""", unsafe_allow_html=True)
