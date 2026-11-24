# Multi-Model PDF RAG Assistant on Amazon Bedrock

A Retrieval-Augmented Generation (RAG) application built with **Amazon Bedrock**, **LangChain**, **FAISS**, and **Streamlit** that enables semantic search and multi-LLM question answering over PDF documents.

## Architecture

```text
PDF Documents (data/)
       │
       ▼
PyPDFDirectoryLoader ──► RecursiveCharacterTextSplitter (chunk_size=10000, overlap=1000)
       │
       ▼
Amazon Titan Text Embeddings v2 (amazon.titan-embed-text-v2:0)
       │
       ▼
Local FAISS Vector Store (faiss_index/)
       │
       ▼
User Query ──► Similarity Search (Top-K = 3) ──► LangChain RetrievalQA
                                                        │
                        ┌───────────────────────────────┴───────────────────────────────┐
                        ▼                                                               ▼
        Anthropic Claude Opus 4.7                                                DeepSeek V3.2
     (via Bedrock Converse API)                                           (via Bedrock Converse API)
```

## Key Features
* **Serverless Foundation Models:** Integrates **Amazon Titan Text Embeddings v2** for dense vector generation and **ChatBedrockConverse** for unified multi-model inference across Anthropic Claude and DeepSeek models.
* **Persistent Local Vector Index:** Stores and reloads document embeddings locally with **FAISS** (`faiss_index/`) to avoid recomputing vectors on every query.
* **Configurable Chunking & Retrieval:** Splits multi-page PDFs with `RecursiveCharacterTextSplitter` and retrieves the top-$k$ ($k=3$) semantically relevant chunks for grounded, low-hallucination responses.
* **Interactive Streamlit UI:** Provides one-click vector index rebuilds from the sidebar and side-by-side model comparison from the main workspace.

## Project Structure

```text
├── data/                  # Input PDF documents for ingestion
├── app.py                 # Main Streamlit RAG application
├── novapro.py             # Standalone Amazon Nova Pro Bedrock inference script
├── requirements.txt       # Python dependencies
├── .gitignore             # Excludes venv, local FAISS index, and IDE artifacts
└── README.md              # Project documentation
```

## Tech Stack
* **Cloud & AI:** Amazon Bedrock (`amazon.titan-embed-text-v2:0`, `anthropic.claude-opus-4-7`, `deepseek.v3.2`), AWS SDK (`boto3`)
* **Orchestration:** LangChain (`langchain-aws`, `langchain-community`, `langchain-text-splitters`, `langchain-classic`)
* **Vector Database:** FAISS (`faiss-cpu`)
* **Frontend:** Streamlit
* **Language:** Python 3.10+

## Prerequisites
* **Python 3.10+**
* **AWS Account** with Amazon Bedrock model access enabled in your target region (default: `ap-southeast-2`)
* **AWS CLI / IAM Credentials** configured locally with `bedrock:InvokeModel` permissions

## Quickstart

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)<your-username>/amazon-bedrock-pdf-rag.git
cd amazon-bedrock-pdf-rag
```

### 2. Create & Activate a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials
```bash
aws configure
```
Ensure your configured AWS region matches `region_name` in `app.py` (default: `ap-southeast-2`).

### 5. Run the Application
```bash
python -m streamlit run app.py
```
1. Place your PDF files inside the `data/` directory.
2. Click **Vectors Update** in the sidebar to generate and save the FAISS vector index.
3. Type a question in the input box and click **Claude LLM Response** or **Deepseek LLM Response**.

## Security & Best Practices
* **Zero Hardcoded Secrets:** Authentication uses `boto3`'s standard credential chain (`~/.aws/credentials` or IAM roles).
* **Clean Repository Hygiene:** Virtual environments (`venv/`), IDE settings (`.vscode/`), and generated local vector indexes (`faiss_index/`) are excluded via `.gitignore`.