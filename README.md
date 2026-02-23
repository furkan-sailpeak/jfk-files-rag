# JFK Files Research System

A Retrieval-Augmented Generation (RAG) system for querying declassified JFK assassination documents. Built as part of a Master of Statistics & Data Science thesis at KU Leuven.

> *"Topic Modeling and Thematic Analysis of JFK Assassination Files Using NLP"*

## Features

- **Document Q&A** — Ask questions about the JFK files and get structured research reports with in-text citations linked directly to source PDFs
- **Conversation Memory** — Follow-up questions maintain context from previous exchanges
- **Source Verification** — Every claim is cited with clickable references that open the original document at the exact page on the National Archives
- **Archive Statistics** — Real-time overview of the document collection (pages, redactions, handwriting, stamps)
- **Entity Extraction & Summarization** — Extract names or summarize content from retrieved documents

## Architecture

- **Backend**: Python, Flask, PostgreSQL (Supabase), Groq (LLaMA 3.3 70B)
- **Frontend**: React, Vite, Framer Motion, Lucide Icons
- **Database**: ~70,000+ OCR-processed pages from the 2025 JFK document release
- **Deployment**: Docker / Railway

The system searches a PostgreSQL database of OCR-processed JFK document pages, retrieves relevant context, and generates structured research reports using an LLM. Source documents link directly to the National Archives via Google Cloud Storage.

### Backend Flow (`app.py`)

```
User Query
    │
    ▼
┌─────────────────────────┐
│  Document ID Detection   │──── regex match (e.g., 104-10433-10209)
│  (e.g., 104-XXXXX-XXXXX)│     │
└─────────┬───────────────┘     ▼
          │ no match       ┌──────────────────┐
          ▼                │ Retrieve all     │
┌─────────────────────┐    │ pages of that    │
│  Query Analysis     │    │ specific document│
│  (LLM: keyword      │    └────────┬─────────┘
│   extraction + type) │             │
└─────────┬───────────┘             │
          ▼                         │
┌─────────────────────┐             │
│  PostgreSQL Search   │             │
│  (ILIKE + ranking)   │             │
│  Supabase / jfk_pages│             │
└─────────┬───────────┘             │
          ▼                         ▼
┌─────────────────────────────────────┐
│  Build Numbered Context             │
│  [1] Source: file.pdf, Page N       │
│  [2] Source: file.pdf, Page M       │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  LLM Generation (Groq/LLaMA 3.3)   │
│  System prompt + conversation       │
│  history + retrieved documents      │
│  → Structured report with [N] cites │
└─────────────────┬───────────────────┘
                  ▼
┌─────────────────────────────────────┐
│  Response to Frontend               │
│  { answer, sources[], query_type }  │
│                                     │
│  Frontend: [N] → clickable links    │
│  to NARA PDFs (GCS) at exact page   │
└─────────────────────────────────────┘
```

## Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- A Supabase PostgreSQL database with the `jfk_pages` table
- A [Groq](https://console.groq.com/) API key

### Environment Variables

Copy the example and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string (Supabase) |
| `GROQ_API_KEY` | API key for Groq LLM |

### Local Development

```bash
# Backend
cd rag
pip install -r requirements.txt
python app.py

# Frontend (separate terminal)
cd rag/frontend
npm install
npm run dev
```

The Vite dev server proxies `/api` requests to Flask on port 5001.

### Docker

```bash
cd rag
docker build -t jfk-rag .
docker run -p 5001:5001 \
  -e DATABASE_URL="your_connection_string" \
  -e GROQ_API_KEY="your_api_key" \
  jfk-rag
```

### Railway Deployment

1. Set the root directory to `rag/`
2. Add `DATABASE_URL` and `GROQ_API_KEY` as environment variables
3. Railway auto-detects the Dockerfile

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/api/stats` | GET | Archive statistics |
| `/api/chat` | POST | Query documents with RAG (supports conversation history) |
| `/api/analyze` | POST | Extract names or summarize text |
| `/api/pdf/<filename>` | GET | Redirect to NARA archive PDF |

## Project Structure

```
rag/
├── app.py                  # Flask backend (RAG logic, PostgreSQL, API routes)
├── requirements.txt        # Python dependencies
├── Dockerfile              # Multi-stage build for Railway
├── .dockerignore
└── frontend/
    ├── src/
    │   ├── App.jsx         # React app with citation linking
    │   ├── index.css       # Archival/classified document theme
    │   └── main.jsx
    ├── index.html
    ├── vite.config.js      # Dev proxy + build config
    └── package.json
```

## License

This project is part of academic research at KU Leuven. The JFK documents are public domain, released by the National Archives and Records Administration (NARA).
