"""
services/ingest.py — Document ingestion pipeline.

Pipeline:
  PubChem CID / raw text
      → fetch / receive
      → chunk (overlapping)
      → embed (sentence-transformers)
      → index (FAISS)
      → persist (disk)

Also contains sample chemistry documents for demo purposes.
"""
from __future__ import annotations
import uuid
import requests
from loguru import logger

from services.chunker import chunk_document, Chunk
from services.embedder import get_embedder
from services.vectorstore import get_vectorstore


# ── PubChem fetcher ───────────────────────────────────────────────────────────

def fetch_pubchem_description(cid: int) -> dict | None:
    """Fetch compound description from PubChem REST API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        descriptions = data.get("InformationList", {}).get("Information", [])
        if not descriptions:
            return None

        texts = []
        title = f"PubChem CID {cid}"
        for info in descriptions:
            if "Title" in info:
                title = info["Title"]
            if "Description" in info:
                texts.append(info["Description"])

        if not texts:
            return None

        return {
            "doc_id": f"pubchem_{cid}",
            "title":  title,
            "text":   " ".join(texts),
            "source": f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}",
            "metadata": {"cid": cid, "source_type": "pubchem"},
        }
    except Exception as e:
        logger.warning("Failed to fetch PubChem CID {}: {}", cid, e)
        return None


def fetch_pubchem_properties(cid: int) -> str:
    """Fetch key chemical properties as text."""
    props = ["MolecularFormula", "MolecularWeight", "IUPACName",
             "CanonicalSMILES", "XLogP", "TPSA"]
    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}"
           f"/property/{','.join(props)}/JSON")
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        props_data = data.get("PropertyTable", {}).get("Properties", [{}])[0]

        parts = []
        if "IUPACName" in props_data:
            parts.append(f"IUPAC name: {props_data['IUPACName']}.")
        if "MolecularFormula" in props_data:
            parts.append(f"Molecular formula: {props_data['MolecularFormula']}.")
        if "MolecularWeight" in props_data:
            parts.append(f"Molecular weight: {props_data['MolecularWeight']} g/mol.")
        if "CanonicalSMILES" in props_data:
            parts.append(f"SMILES: {props_data['CanonicalSMILES']}.")
        if "XLogP" in props_data:
            parts.append(f"LogP (lipophilicity): {props_data['XLogP']}.")
        if "TPSA" in props_data:
            parts.append(f"Topological polar surface area: {props_data['TPSA']} Å².")
        return " ".join(parts)
    except Exception:
        return ""


# ── Core ingestion logic ──────────────────────────────────────────────────────

def ingest_document(
    doc_id: str,
    title: str,
    text: str,
    source: str,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Chunk, embed, and index a single document."""
    embedder = get_embedder()
    store    = get_vectorstore()

    # Skip if already indexed
    if doc_id in store.doc_ids():
        logger.info("Already indexed, skipping: {}", doc_id)
        return []

    # Chunk
    chunks = chunk_document(doc_id, title, text, source, metadata)
    if not chunks:
        logger.warning("No chunks produced for: {}", title)
        return []

    # Embed
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts)

    # Index
    store.add(chunks, embeddings)
    logger.info("Ingested '{}' | {} chunks", title, len(chunks))
    return chunks


def ingest_pubchem_cids(cids: list[int]) -> dict:
    """Ingest a list of PubChem compound IDs."""
    ingested = 0
    total_chunks = 0

    for cid in cids:
        doc = fetch_pubchem_description(cid)
        if doc is None:
            continue

        # Enrich with properties
        props_text = fetch_pubchem_properties(cid)
        if props_text:
            doc["text"] = props_text + " " + doc["text"]

        chunks = ingest_document(
            doc_id=doc["doc_id"],
            title=doc["title"],
            text=doc["text"],
            source=doc["source"],
            metadata=doc["metadata"],
        )
        if chunks:
            ingested += 1
            total_chunks += len(chunks)

    return {
        "ingested": ingested,
        "total_chunks": total_chunks,
        "total_documents": get_vectorstore().count(),
    }


# ── Sample chemistry documents for demo ───────────────────────────────────────

SAMPLE_DOCUMENTS = [
    {
        "title": "Aqueous Solubility in Drug Discovery",
        "source": "ChemRAG Demo",
        "text": """
Aqueous solubility is one of the most critical physicochemical properties in drug discovery.
Poor solubility is a major cause of drug attrition, with estimates suggesting that 40% of
new chemical entities suffer from low aqueous solubility. Aqueous solubility is defined as
the maximum amount of a substance that dissolves in a given volume of water at a specific
temperature and pH. It is typically expressed in mg/mL, μg/mL, or as the log of molar
solubility (LogS). Factors affecting aqueous solubility include crystal packing energy,
melting point, lipophilicity (LogP), hydrogen bonding capacity, molecular weight, and polar
surface area (PSA). The rule of five by Lipinski states that poor absorption or permeability
is more likely when molecular weight is over 500 Da, LogP is over 5, hydrogen bond donors
exceed 5, and hydrogen bond acceptors exceed 10. Computational methods for solubility
prediction include quantitative structure-property relationship (QSPR) models, machine
learning approaches using molecular fingerprints, and deep learning with graph neural
networks (GNNs). The ESOL (estimated solubility) model by Delaney uses a linear regression
on molecular descriptors. Modern approaches use ChemBERT and MPNN architectures trained on
curated datasets from PubChem and ChEMBL. Temperature significantly affects solubility —
most organic compounds show increased solubility with rising temperature. pH affects the
solubility of ionisable compounds through the Henderson-Hasselbalch equation.
        """.strip(),
    },
    {
        "title": "Retrosynthesis and Computer-Aided Synthesis Planning",
        "source": "ChemRAG Demo",
        "text": """
Retrosynthesis is a problem-solving technique in organic chemistry for planning the synthesis
of complex molecules. Introduced by E.J. Corey in the 1960s, it involves working backwards
from the target molecule to identify simpler precursor molecules through a series of
retrosynthetic steps. Computer-aided synthesis planning (CASP) has been revolutionised by
machine learning. Template-based methods use reaction rules extracted from databases such as
Reaxys and USPTO to predict transformations. Template-free methods use sequence-to-sequence
neural networks that treat SMILES strings as sequences, similar to machine translation.
The USPTO-50K dataset, containing 50,000 patent reactions across 10 reaction classes, is the
standard benchmark for retrosynthesis prediction. State-of-the-art models achieve over 85%
Top-1 accuracy on this benchmark. Reaction classes in USPTO-50K include heteroatom alkylation
and arylation, acylation, C-C bond formation, heterocycle formation, protections,
deprotections, reductions, oxidations, functional group interconversion (FGI), and functional
group addition (FGA). Beam search is used during inference to generate multiple candidate
reactant predictions, ranked by log-probability score. SMILES augmentation, which generates
multiple valid SMILES representations of the same molecule, is a key technique for improving
model generalisation. Reaxys Predictive Retrosynthesis integrates AI-powered retrosynthesis
with the world's largest curated reaction database.
        """.strip(),
    },
    {
        "title": "Chemical Named Entity Recognition in Scientific Literature",
        "source": "ChemRAG Demo",
        "text": """
Chemical named entity recognition (NER) is the task of identifying and classifying chemical
entities in scientific text. Chemical entities include compound names (IUPAC, trivial, trade),
molecular formulas, SMILES strings, reaction identifiers, protein names, gene names, and
disease terms. Chemical NER is challenging due to the diversity of chemical nomenclature —
a single compound may be referred to by its IUPAC name, a common name, an abbreviation,
a registry number (CAS, PubChem CID), or its SMILES representation. Domain-specific
transformer models such as ChemBERT, MatBERT, and BioBERT are pre-trained on large chemistry
and biomedical corpora and significantly outperform general-purpose BERT on chemical NER.
Common annotation schemes include BIO (Beginning, Inside, Outside) tagging and BIOES
(Beginning, Inside, Other, End, Single) tagging. Evaluation metrics are entity-level
precision, recall, and F1 score. Key datasets for chemical NER include BC5CDR (chemicals
and diseases), CHEMDNER (chemical compound and drug names), and NLMChem. Information
extraction pipelines for databases like Reaxys and Embase use NER as the first step in a
pipeline that also includes relation extraction (identifying reactions between entities),
entity linking (normalising extracted entities to database identifiers), and attribute
extraction (extracting properties like yield, temperature, solvent). Hugging Face Transformers
provides the standard framework for fine-tuning chemical NER models.
        """.strip(),
    },
    {
        "title": "Transformer Models for Chemistry — ChemBERT and Molecular Transformers",
        "source": "ChemRAG Demo",
        "text": """
Transformer models have transformed natural language processing and are increasingly applied
to chemistry. BERT (Bidirectional Encoder Representations from Transformers) uses a
self-attention mechanism to encode contextual representations of tokens. For chemistry,
SMILES strings are treated as sequences of tokens, where each atom, bond, and bracket is a
token. ChemBERT is a BERT model pre-trained on 77 million SMILES strings from PubChem.
It learns molecular representations that encode chemical structure, functional groups, and
bonding patterns. Fine-tuned ChemBERT achieves strong performance on molecular property
prediction tasks including solubility, toxicity, and bioactivity. The Molecular Transformer
by Schwaller et al. applies sequence-to-sequence transformer architectures to chemical
reaction prediction, treating reactant and product SMILES as source and target sequences.
This model achieves state-of-the-art performance on reaction prediction benchmarks.
Graph Neural Networks (GNNs) and Message Passing Neural Networks (MPNNs) are complementary
to sequence-based models. GNNs represent molecules as graphs where atoms are nodes and bonds
are edges, allowing message passing to aggregate local chemical environment information.
Multi-modal architectures that combine SMILES-based transformers with GNNs and traditional
molecular fingerprints achieve the best performance on molecular property prediction by
capturing different aspects of molecular structure. Hugging Face provides pre-trained
ChemBERT models and the transformers library for fine-tuning on downstream tasks.
        """.strip(),
    },
    {
        "title": "Retrieval-Augmented Generation for Scientific Question Answering",
        "source": "ChemRAG Demo",
        "text": """
Retrieval-Augmented Generation (RAG) combines information retrieval with large language model
generation to produce grounded, citation-backed answers. In a RAG system, a user query is
first used to retrieve relevant documents from a knowledge base using dense vector similarity
search. The retrieved documents are then provided as context to a language model, which
generates an answer grounded in the retrieved evidence. RAG addresses the key limitation of
parametric language models — their tendency to hallucinate facts — by anchoring generation
to retrieved documents. The RAG pipeline consists of four main components. First, document
ingestion: documents are chunked into passages of typically 200-500 tokens with overlap,
then embedded into dense vectors using a sentence encoder. Second, indexing: embeddings are
stored in a vector database such as FAISS, Pinecone, or OpenSearch for efficient nearest
neighbour retrieval. Third, retrieval: the user query is encoded and top-k most similar
document chunks are retrieved by cosine similarity. Fourth, generation: retrieved chunks are
concatenated into a context window and passed to the LLM with the query. Evaluation of RAG
systems uses metrics including context relevance (are the retrieved documents relevant?),
faithfulness (does the answer stay within the retrieved context?), and answer relevance (does
the answer address the question?). The RAGAS framework provides automated evaluation of
these metrics. For scientific and chemistry RAG, domain-specific embedding models like
ChemBERT or SciBERT improve retrieval quality over general-purpose models.
        """.strip(),
    },
]


def ingest_sample_documents() -> dict:
    """Ingest the built-in sample chemistry documents."""
    ingested = 0
    total_chunks = 0

    for doc in SAMPLE_DOCUMENTS:
        doc_id = "sample_" + doc["title"].lower().replace(" ", "_")[:40]
        chunks = ingest_document(
            doc_id=doc_id,
            title=doc["title"],
            text=doc["text"],
            source=doc["source"],
            metadata={"source_type": "sample"},
        )
        if chunks:
            ingested += 1
            total_chunks += len(chunks)

    return {
        "ingested": ingested,
        "total_chunks": total_chunks,
        "total_documents": get_vectorstore().count(),
    }
