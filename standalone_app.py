"""
standalone_app.py — ChemRAG Standalone for Streamlit Cloud
Uses Groq for LLM + simple TF-IDF retrieval (no heavy ML models)
Falls back gracefully at every step with clear error messages.
"""
import os, re, uuid, math, requests
from collections import defaultdict
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(
    page_title="ChemRAG — Chemical Literature Assistant",
    page_icon="🧪", layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.hero {
    background: linear-gradient(135deg,#0a1628,#1a2e4a,#0d4f3c);
    padding:28px 36px; border-radius:16px; color:white; margin-bottom:20px;
}
.hero h1 { font-size:1.9rem; font-weight:700; margin:0 0 6px 0; }
.hero p  { opacity:0.7; margin:0; font-size:0.9rem; }
.step { display:inline-block; padding:5px 12px; border-radius:16px;
        font-size:0.78rem; font-weight:600; margin:3px; }
.s1 { background:rgba(99,179,237,.15); color:#63b3ed; border:1px solid #63b3ed; }
.s2 { background:rgba(72,187,120,.15); color:#48bb78; border:1px solid #48bb78; }
.s3 { background:rgba(237,137,54,.15); color:#ed8936; border:1px solid #ed8936; }
.s4 { background:rgba(159,122,234,.15); color:#9f7aea; border:1px solid #9f7aea; }
.answer-box {
    background:linear-gradient(135deg,rgba(13,79,60,.08),rgba(10,22,40,.08));
    border:1px solid #2d6a4f; border-radius:12px; padding:20px; margin:12px 0;
}
.mbox { background:white; border:1px solid #e2e8f0; border-radius:10px;
        padding:16px; text-align:center; }
.mnum { font-size:1.8rem; font-weight:700; color:#2d6a4f; }
.mlbl { font-size:0.75rem; color:#718096; margin-top:4px; }
section[data-testid="stSidebar"] { background:#0a1628; }
section[data-testid="stSidebar"] * { color:#e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)


# ── Groq key ──────────────────────────────────────────────────────────────────
def get_groq_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.environ.get("GROQ_API_KEY", "")


# ── TF-IDF retrieval (no heavy ML needed) ────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

def build_tfidf(chunks):
    """Build TF-IDF index from list of chunk dicts."""
    N = len(chunks)
    df = defaultdict(int)
    chunk_tfs = []
    for chunk in chunks:
        words = tokenize(chunk["text"])
        tf = defaultdict(int)
        for w in words:
            tf[w] += 1
        total = max(len(words), 1)
        tf = {w: c/total for w, c in tf.items()}
        chunk_tfs.append(tf)
        for w in tf:
            df[w] += 1

    idf = {w: math.log((N+1)/(df[w]+1))+1 for w in df}
    return chunk_tfs, idf

def score_chunk(tf, idf, query_words):
    return sum(tf.get(w, 0) * idf.get(w, 0) for w in query_words)

def retrieve(query, chunks, chunk_tfs, idf, top_k=5):
    q_words = tokenize(query)
    scores = [score_chunk(tf, idf, q_words) for tf in chunk_tfs]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    seen = set()
    for idx, score in ranked:
        if score > 0 and chunks[idx]["doc_id"] not in seen:
            seen.add(chunks[idx]["doc_id"])
            results.append((chunks[idx], round(score, 4)))
        if len(results) >= top_k:
            break
    return results


# ── Document store (session state) ───────────────────────────────────────────
def init_store():
    if "chunks" not in st.session_state:
        st.session_state["chunks"]     = []
        st.session_state["chunk_tfs"]  = []
        st.session_state["idf"]        = {}
        st.session_state["doc_ids"]    = set()

def get_chunks():     return st.session_state.get("chunks", [])
def get_chunk_tfs():  return st.session_state.get("chunk_tfs", [])
def get_idf():        return st.session_state.get("idf", {})
def get_doc_ids():    return st.session_state.get("doc_ids", set())

def rebuild_index():
    chunks = get_chunks()
    if chunks:
        tfs, idf = build_tfidf(chunks)
        st.session_state["chunk_tfs"] = tfs
        st.session_state["idf"]       = idf

def add_document(doc_id, title, text, source, metadata=None):
    init_store()
    if doc_id in get_doc_ids():
        return 0
    # Chunk text
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    raw_chunks, current, size = [], [], 0
    for sent in sentences:
        words = sent.split()
        if size + len(words) > 400 and current:
            raw_chunks.append(" ".join(current))
            current = current[-50:] + words
            size = len(current)
        else:
            current.extend(words)
            size += len(words)
    if current:
        raw_chunks.append(" ".join(current))

    for i, chunk_text in enumerate(raw_chunks):
        st.session_state["chunks"].append({
            "chunk_id":    str(uuid.uuid4()),
            "doc_id":      doc_id,
            "title":       title,
            "source":      source,
            "text":        chunk_text,
            "chunk_index": i,
            "metadata":    metadata or {},
        })
    st.session_state["doc_ids"].add(doc_id)
    rebuild_index()
    return len(raw_chunks)

def get_documents():
    docs = {}
    for c in get_chunks():
        did = c["doc_id"]
        if did not in docs:
            docs[did] = {"doc_id": did, "title": c["title"],
                         "source": c["source"], "chunk_count": 0,
                         "preview": c["text"][:200]}
        docs[did]["chunk_count"] += 1
    return list(docs.values())

def chunk_count():  return len(get_chunks())
def doc_count():    return len(get_doc_ids())


# ── Groq LLM ──────────────────────────────────────────────────────────────────
SYSTEM = """You are ChemRAG, an expert chemistry research assistant.
Answer ONLY from the provided context documents.
If the context is insufficient, say so clearly.
Always cite which document(s) your answer comes from.
Do not invent chemical facts or data. Be precise and scientific."""

def call_groq(question, context):
    key = get_groq_key()
    if not key:
        return "⚠️ No GROQ_API_KEY set. Add it in Streamlit Secrets → Settings → Secrets."

    prompt = f"""Context documents:
{context}

---
Question: {question}

Answer based only on the context above. Cite the relevant document title(s)."""

    try:
        import json as _json
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "temperature": 0.1,
            "max_tokens": 600,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user",   "content": prompt},
            ]
        }
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Groq error: {e}"


# ── RAG pipeline ──────────────────────────────────────────────────────────────
def run_rag(question, top_k=5):
    chunks = get_chunks()
    if not chunks:
        return {"answer": "No documents indexed yet. Go to **📥 Ingest** and click **Load Sample Documents**.",
                "sources": [], "retrieval_count": 0, "grounded": False}

    retrieved = retrieve(question, chunks, get_chunk_tfs(), get_idf(), top_k)
    if not retrieved:
        return {"answer": "No relevant documents found for your question.",
                "sources": [], "retrieval_count": 0, "grounded": False}

    context = "\n\n".join(
        f"[Document {i+1}: {c['title']}]\n{c['text']}\n(Source: {c['source']})"
        for i, (c, _) in enumerate(retrieved)
    )
    answer = call_groq(question, context)

    sources, seen = [], set()
    for chunk, score in retrieved:
        if chunk["doc_id"] not in seen:
            seen.add(chunk["doc_id"])
            sources.append({"title": chunk["title"], "source": chunk["source"],
                            "excerpt": chunk["text"][:300]+"...", "score": score})

    return {"answer": answer, "sources": sources,
            "retrieval_count": len(retrieved), "grounded": True}


# ── Sample documents ──────────────────────────────────────────────────────────
SAMPLES = [
    ("Aqueous Solubility in Drug Discovery", "ChemRAG Demo", """
Aqueous solubility is one of the most critical physicochemical properties in drug discovery.
Poor solubility causes 40% of new chemical entities to fail. Aqueous solubility is the maximum
amount dissolved in water at given temperature and pH, expressed as LogS or mg/mL. Factors include
lipophilicity (LogP), hydrogen bonding, molecular weight, and polar surface area (PSA). Lipinski's
rule of five: molecular weight under 500 Da, LogP under 5, hydrogen bond donors under 5, acceptors
under 10. High LogP means high lipophilicity and poor water solubility. Machine learning models
using ChemBERT and MPNN architectures predict solubility with RMSE below 1 log unit. The ESOL
model by Delaney uses linear regression on LogP, molecular weight, and rotatable bonds. Temperature
and pH significantly affect ionisable compounds via the Henderson-Hasselbalch equation. PubChem
and ChEMBL provide large training datasets for solubility prediction models.
    """.strip()),
    ("Retrosynthesis and Computer-Aided Synthesis Planning", "ChemRAG Demo", """
Retrosynthesis introduced by E.J. Corey works backwards from target molecule to simpler precursors.
Computer-aided synthesis planning uses machine learning with template-free seq2seq neural networks
treating SMILES as sequences. The USPTO-50K dataset with 50,000 reactions across 10 classes is
the standard benchmark — state-of-the-art models achieve over 85% Top-1 accuracy. Reaction classes:
heteroatom alkylation, acylation, C-C bond formation, heterocycle formation, protections,
deprotections, reductions, oxidations, FGI, and FGA. Beam search generates candidate reactants
ranked by log-probability score. SMILES augmentation generates multiple valid SMILES per molecule
improving generalisation by 8-10%. Reaxys Predictive Retrosynthesis integrates AI with over 500
million curated reactions. The transformer seq2seq model by Schwaller achieves top benchmark scores
treating chemical SMILES like natural language translation sentences.
    """.strip()),
    ("Chemical Named Entity Recognition and Information Extraction", "ChemRAG Demo", """
Chemical NER identifies compound names, formulas, SMILES strings, protein names in scientific text.
Challenges include diverse nomenclature — IUPAC name, common name, abbreviation, CAS, PubChem CID.
ChemBERT, MatBERT, BioBERT outperform general BERT on chemistry text significantly. BIO tagging:
Beginning, Inside, Outside labels for token classification. Key datasets: BC5CDR, CHEMDNER, NLMChem.
Evaluation: entity-level precision, recall, F1 score per entity type. Elsevier enrichment pipelines
for Reaxys and Embase use NER then relation extraction identifying reactions between entities, entity
linking normalising to database identifiers, attribute extraction for yield temperature solvent.
Hugging Face Transformers provides fine-tuning framework for domain-specific NER models. Active
learning from expert corrections improves model quality over time continuously.
    """.strip()),
    ("ChemBERT and Transformer Models for Molecular Property Prediction", "ChemRAG Demo", """
ChemBERT is BERT pre-trained on 77 million SMILES strings from PubChem learning molecular
representations encoding structure bonding and functional groups. SMILES tokens: each atom bond
bracket is one token. Fine-tuned ChemBERT achieves strong performance on solubility toxicity
bioactivity prediction tasks. The Molecular Transformer by Schwaller applies seq2seq transformer
to reaction prediction treating SMILES as source and target sequences achieving state-of-the-art
on USPTO benchmarks. Graph Neural Networks represent molecules as graphs with atoms as nodes and
bonds as edges enabling message passing to aggregate local chemical environment. MPNNs (Message
Passing Neural Networks) iterate over graph neighbourhood to build molecular representations.
Multi-modal architectures combining SMILES transformers with GNNs and molecular fingerprints achieve
best performance by capturing complementary aspects of structure. Hugging Face provides pre-trained
ChemBERT models for downstream fine-tuning tasks.
    """.strip()),
    ("RAG — Retrieval-Augmented Generation for Scientific Question Answering", "ChemRAG Demo", """
RAG combines information retrieval with language model generation for grounded citation-backed answers.
Pipeline: document ingestion chunks text into 200-500 token passages with overlap, embeds using
sentence encoder. Indexing stores embeddings in FAISS Pinecone or OpenSearch vector database.
Retrieval encodes user query and finds top-k similar chunks by cosine similarity. Generation passes
retrieved chunks as context to LLM with query. RAG prevents hallucination by anchoring generation
to retrieved documents. Evaluation: context relevance (retrieved docs relevant?), faithfulness
(answer within context?), answer relevance (addresses question?). RAGAS framework automates
evaluation. FAISS IndexFlatIP uses inner product equivalent to cosine similarity on normalised vectors.
For production: IndexIVFFlat faster at million-scale, IndexHNSW approximate nearest neighbour.
TF-IDF is simpler retrieval: term frequency times inverse document frequency scores relevance.
    """.strip()),
]

def ingest_samples():
    init_store()
    total_chunks = 0
    for title, source, text in SAMPLES:
        doc_id = "sample_" + title.lower().replace(" ","_")[:40]
        n = add_document(doc_id, title, text, source)
        total_chunks += n
    return len(SAMPLES), total_chunks


EXAMPLES = [
    "What is the relationship between LogP and drug solubility?",
    "How does retrosynthesis work and what is USPTO-50K?",
    "Explain how ChemBERT represents molecular structures",
    "What is FAISS and how does RAG prevent hallucination?",
    "What reaction classes are in the USPTO-50K dataset?",
    "How does SMILES augmentation improve model performance?",
    "What is beam search in retrosynthesis prediction?",
    "How do GNNs represent molecules differently from SMILES?",
    "What NLP challenges are specific to chemistry text?",
    "Explain Lipinski rule of five for drug discovery",
]


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
init_store()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 8px'>
        <div style='font-size:2.4rem'>🧪</div>
        <div style='font-size:1.05rem;font-weight:700'>ChemRAG</div>
        <div style='font-size:0.72rem;opacity:0.5'>Chemical Literature Assistant</div>
        <div style='font-size:0.68rem;opacity:0.4;margin-top:2px'>Dr. Mushtaq Ali · KIT</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    page = st.radio("", [
        "🏠 Home", "💬 Ask ChemRAG",
        "📥 Ingest", "📚 Library", "🔬 How It Works"
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"**📄 Docs:** {doc_count()}  |  **🧩 Chunks:** {chunk_count()}")
    groq_ok = bool(get_groq_key())
    st.markdown("🟢 **Groq ready**" if groq_ok else "🔴 **No Groq key**")
    if chunk_count() == 0:
        st.warning("⚠️ Go to Ingest first")
    st.divider()
    top_k = st.slider("Retrieve k chunks", 1, 10, 5)


# ─────────────────────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Home":
    st.markdown("""
    <div class="hero">
        <h1>🧪 ChemRAG — Chemical Literature RAG Assistant</h1>
        <p>Retrieval-Augmented Generation over chemistry papers and PubChem data · Dr. Mushtaq Ali · KIT</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin:12px 0'>
        <span class="step s1">① TF-IDF Retrieval</span>→
        <span class="step s2">② Context Assembly</span>→
        <span class="step s3">③ Prompt Engineering</span>→
        <span class="step s4">④ Groq LLM</span>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col, num, lbl in [
        (c1, doc_count(), "Documents"),
        (c2, chunk_count(), "Indexed Chunks"),
        (c3, "TF-IDF", "Retrieval"),
        (c4, "Groq", "LLM"),
    ]:
        col.markdown(f'<div class="mbox"><div class="mnum">{num}</div><div class="mlbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.write("")
    cl, cr = st.columns(2)
    with cl:
        st.subheader("🚀 Quick Start")
        st.markdown("""
        1. Go to **📥 Ingest** → **Load Sample Docs**
        2. Go to **💬 Ask ChemRAG**
        3. Type a chemistry question → get grounded answer
        """)
    with cr:
        st.subheader("Try these questions")
        for ex in EXAMPLES[:5]:
            if st.button(f"→ {ex[:55]}", key=f"h_{ex[:15]}", use_container_width=True):
                st.session_state["pq"] = ex
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# ASK
# ─────────────────────────────────────────────────────────────────────────────
elif page == "💬 Ask ChemRAG":
    st.header("💬 Ask ChemRAG")
    st.caption("Ask any chemistry question — answered from indexed documents with citations.")

    default_q = st.session_state.pop("pq", "")
    question  = st.text_input("Your chemistry question", value=default_q,
        placeholder="e.g. What is the relationship between LogP and drug solubility?")

    ca, cb = st.columns([3,1])
    with ca:
        ask = st.button("🔍 Ask ChemRAG", type="primary", use_container_width=True)
    with cb:
        if st.button("Clear", use_container_width=True):
            st.session_state.pop("result", None)
            st.rerun()

    st.caption("Quick examples:")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES[:6]):
        with cols[i%3]:
            short = ex[:42]+"…" if len(ex)>42 else ex
            if st.button(short, key=f"e_{i}", use_container_width=True):
                st.session_state["pq"] = ex
                st.rerun()

    if ask and question.strip():
        with st.spinner("🔍 Retrieving relevant chunks... 💭 Generating answer..."):
            st.session_state["result"] = run_rag(question.strip(), top_k=top_k)

    result = st.session_state.get("result")
    if not result:
        st.stop()

    st.divider()
    grounded = result.get("grounded", False)
    st.markdown(f"""
    <div style='display:flex;gap:8px;align-items:center;margin-bottom:8px'>
        <span class="step {'s2' if grounded else 's1'}">
            {'✅ Grounded answer' if grounded else '⚠️ Issue'}
        </span>
        <span style='font-size:0.78rem;color:#718096'>
            {result['retrieval_count']} chunks retrieved · llama-3.1-8b-instant
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
            fig.update_layout(title="TF-IDF Retrieval Scores",
                height=max(180, len(sources)*45),
                margin=dict(l=0,r=60,t=40,b=20), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        for s in sources:
            with st.expander(f"📄 {s['title']}  (score: {s['score']:.4f})", expanded=False):
                st.markdown(f"**Source:** {s['source']}")
                st.markdown(s["excerpt"])


# ─────────────────────────────────────────────────────────────────────────────
# INGEST
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📥 Ingest":
    st.header("📥 Ingest Documents")
    tab1, tab2, tab3, tab4 = st.tabs(["📦 Sample Docs", "🔬 PubChem CIDs", "📝 Paste Text", "📄 Upload PDF"])

    with tab1:
        st.markdown("Load 5 built-in chemistry documents: solubility, retrosynthesis, NER, ChemBERT, RAG.")
        if st.button("⚡ Load Sample Documents", type="primary"):
            with st.spinner("Ingesting 5 chemistry documents..."):
                n_docs, n_chunks = ingest_samples()
            st.success(f"✅ Ingested {n_docs} documents ({n_chunks} chunks)")
            st.rerun()

    with tab2:
        cid_input = st.text_area("PubChem CIDs (comma separated)",
            value="2244, 5090, 4091, 3672",
            help="2244=Aspirin, 5090=Caffeine, 4091=Paracetamol, 3672=Ibuprofen")
        if st.button("🔬 Fetch from PubChem", type="primary"):
            raw  = cid_input.replace("\n",",").split(",")
            cids = [int(x.strip()) for x in raw if x.strip().isdigit()]
            if not cids:
                st.error("No valid CIDs")
            else:
                total = 0
                prog  = st.progress(0)
                for i, cid in enumerate(cids):
                    try:
                        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
                        r   = requests.get(url, timeout=10)
                        r.raise_for_status()
                        info = r.json().get("InformationList",{}).get("Information",[])
                        title, texts = f"PubChem CID {cid}", []
                        for item in info:
                            if "Title" in item: title = item["Title"]
                            if "Description" in item: texts.append(item["Description"])
                        if texts:
                            n = add_document(f"pubchem_{cid}", title,
                                             " ".join(texts),
                                             f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}")
                            total += n
                    except Exception as e:
                        st.warning(f"CID {cid} failed: {e}")
                    prog.progress((i+1)/len(cids))
                st.success(f"✅ Ingested {len(cids)} compounds ({total} chunks)")
                st.rerun()

    with tab3:
        title  = st.text_input("Title", placeholder="Review of SMILES Notation")
        source = st.text_input("Source", value="manual")
        text   = st.text_area("Text", height=200,
            placeholder="Paste chemistry paper abstract or text here...")
        if st.button("📝 Ingest Text", type="primary"):
            if not title or not text:
                st.error("Title and text are required")
            else:
                n = add_document(f"manual_{uuid.uuid4().hex[:8]}", title, text, source)
                st.success(f"Ingested {title} — {n} chunks")
                st.rerun()

    with tab4:
        st.markdown("""
        Upload any chemistry PDF — research paper, review article, thesis chapter.
        Text is automatically extracted, chunked, and indexed for querying.
        """)

        # Try to import PyMuPDF
        try:
            import fitz
            pymupdf_ok = True
        except ImportError:
            pymupdf_ok = False

        if not pymupdf_ok:
            st.warning("PyMuPDF not installed.")
            st.code("pip install pymupdf", language="bash")
        else:
            uploaded = st.file_uploader(
                "Upload chemistry paper (PDF)",
                type=["pdf"],
                help="Upload any PDF research paper"
            )

            if uploaded is not None:
                st.markdown(f"**File:** {uploaded.name} ({uploaded.size // 1024} KB)")

                pdf_title  = st.text_input(
                    "Paper title",
                    value=uploaded.name.replace(".pdf","").replace("_"," ")
                )
                pdf_source = st.text_input(
                    "DOI or URL",
                    placeholder="https://doi.org/10.1021/..."
                )

                if st.button("Extract and Ingest PDF", type="primary"):
                    with st.spinner("Extracting text from PDF..."):
                        try:
                            import fitz
                            pdf_bytes  = uploaded.read()
                            pdf_doc    = fitz.open(stream=pdf_bytes, filetype="pdf")
                            pages_text = []
                            for page_num in range(len(pdf_doc)):
                                page_text = pdf_doc[page_num].get_text()
                                if page_text.strip():
                                    pages_text.append(page_text)
                            pdf_doc.close()

                            full_text  = chr(10).join(pages_text)
                            full_text  = " ".join(full_text.split())
                            word_count = len(full_text.split())

                            st.info(f"Extracted {len(pages_text)} pages, {word_count:,} words")

                            if word_count < 50:
                                st.error("Very little text extracted — PDF may be image-based/scanned")
                            else:
                                doc_id = f"pdf_{uuid.uuid4().hex[:8]}"
                                n = add_document(
                                    doc_id, pdf_title, full_text,
                                    pdf_source or uploaded.name
                                )
                                st.success(f"Ingested {pdf_title} — {n} chunks indexed")

                                with st.expander("Preview extracted text"):
                                    st.text(full_text[:500] + "...")

                                st.rerun()

                        except Exception as e:
                            st.error(f"PDF extraction failed: {e}")

            st.divider()
            st.markdown("**Once ingested, try these research questions:**")
            for q in [
                "What is the main contribution of this paper?",
                "What datasets were used for training and evaluation?",
                "What machine learning models were compared?",
                "What performance metrics were reported?",
                "What are the limitations acknowledged by the authors?",
                "How does this method compare to state of the art?",
            ]:
                if st.button(f"-> {q}", key=f"rq_{q[:20]}", use_container_width=True):
                    st.session_state["pq"] = q
                    st.info("Go to Ask ChemRAG to ask this question")


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
Text split into overlapping chunks (400 words, 50-word overlap) to avoid losing context at boundaries.
Sentence-aware splitting avoids cutting mid-sentence. Each chunk stored with title, source, and metadata.
In production: sentence-transformers dense embeddings replace TF-IDF for semantic retrieval.
        """),
        ("② TF-IDF Retrieval", """
TF-IDF (Term Frequency × Inverse Document Frequency) scores how relevant each chunk is to the query.
TF = how often a term appears in a chunk. IDF = how rare a term is across all chunks.
Top-k highest scoring chunks retrieved. Fast, interpretable, no GPU needed.
Production upgrade: FAISS IndexFlatIP with sentence-transformer embeddings for semantic similarity.
        """),
        ("③ Context Assembly & Prompt Engineering", """
Retrieved chunks formatted as numbered document blocks with title and source URL.
System prompt instructs LLM: answer only from context, cite titles, flag insufficient information.
Temperature = 0.1 for factual chemistry answers. Max tokens = 600 for concise responses.
        """),
        ("④ LLM Generation via Groq", """
Model: llama-3.1-8b-instant via Groq API — free tier, very fast (200+ tokens/sec).
Direct HTTP call to https://api.groq.com/openai/v1/chat/completions (OpenAI-compatible).
Same code works with GPT-4, Claude, or local Ollama by changing the base URL.
Grounding prevents hallucination: context-only instruction forces factual answers.
        """),
        ("⑤ Production Improvements", """
This demo uses TF-IDF retrieval. Production RAG uses dense vector retrieval:
1. Embed chunks with sentence-transformers/all-MiniLM-L6-v2 (384-dim)
2. Store in FAISS IndexFlatIP (cosine similarity on normalised vectors)
3. For 1M+ vectors: IndexIVFFlat (inverted file) or IndexHNSW (approximate NN)
4. Persistent storage: Pinecone, Qdrant, or Weaviate managed vector databases
Evaluation with RAGAS: faithfulness, answer relevance, context precision metrics.
        """),
    ]:
        with st.expander(title):
            st.markdown(body)

    st.subheader("Tech Stack")
    cols = st.columns(4)
    for col, (name, detail) in zip(cols, [
        ("TF-IDF", "Lightweight retrieval\nNo GPU needed"),
        ("Groq API", "llama-3.1-8b\n200+ tok/sec free"),
        ("Streamlit", "Interactive UI\nDirect deployment"),
        ("PubChem API", "Live compound\ndata ingestion"),
    ]):
        col.markdown(f"""
        <div class="mbox">
            <div style='font-weight:700;color:#2d3748;font-size:0.85rem'>{name}</div>
            <div style='font-size:0.72rem;color:#718096;margin-top:4px;white-space:pre-line'>{detail}</div>
        </div>""", unsafe_allow_html=True)
