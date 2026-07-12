# 📄 RAG Document Q&A — Insurance Benefits Assistant

A Retrieval-Augmented Generation (RAG) pipeline built in **Python + LangChain** that answers natural-language questions over a PDF document. It loads a health-insurance *Summary of Benefits and Coverage (SBC)*, retrieves the most relevant passages, and generates grounded answers using only the retrieved context.

Built to understand RAG end to end — document loading, chunking, embeddings, vector search, and prompt-grounded generation.

---

## How it works

```
PDF → chunk → embed → store (ChromaDB) → retrieve top-k → LLM (grounded) → answer
```

| Stage | Tool | What happens |
|---|---|---|
| **Load** | `PyPDFLoader` | Reads the SBC PDF into pages |
| **Chunk** | `RecursiveCharacterTextSplitter` | Splits into 600-char chunks with 100 overlap |
| **Embed** | `HuggingFaceEmbeddings` (`all-MiniLM-L6-v2`) | Turns chunks into vectors |
| **Store** | `Chroma` | Holds vectors for similarity search |
| **Retrieve** | `.as_retriever(k=3)` | Pulls the 3 most relevant chunks per query |
| **Generate** | `ChatGroq` (`qwen/qwen3-32b`) | Answers using only the retrieved context |

---

## Tech stack

`Python` · `LangChain` · `ChromaDB` · `HuggingFace Embeddings` · `Groq` · `PyPDF`

---

## Setup

```bash
# 1. Install dependencies
pip install langchain langchain-community langchain-chroma langchain-groq \
            langchain-huggingface langchain-text-splitters pypdf python-dotenv

# 2. Add your API key to a .env file
echo "GROQ_API_KEY=your_key_here" > .env

# 3. Place your PDF in the project folder
#    (default: sbc-sample.pdf)

# 4. Run the notebook
jupyter notebook RAG.ipynb
```

---

## Example

```python
question = "Are there separate deductibles for prescription drugs?"
print(chain.invoke(question))
```

```
A: Yes, there is a separate deductible for prescription drug coverage —
   $300 for prescription drugs, on top of the overall plan deductible.
```

---

## Key design choice: grounded answers

The system prompt instructs the model to answer **only** from retrieved context, so it doesn't hallucinate facts that aren't in the document. This is the core of a trustworthy RAG system — the answer is traceable back to the source text.

---

## What I learned

- **Chunk size matters** — too small loses context, too large dilutes retrieval relevance.
- **Retrieval quality drives answer quality** — a wrong answer usually means the wrong chunk was retrieved, not a bad LLM.
- **Grounding is a prompt-engineering problem** — tuning the system prompt to the document type (a cost-comparison table) sharpened accuracy.

---

## Roadmap

- [ ] Add source citations (which chunk / page an answer came from)
- [ ] Swap in a re-ranker for better retrieval precision
- [ ] Add evaluation (RAGAS) to measure retrieval + answer quality
- [ ] Rebuild the orchestration in LangGraph
