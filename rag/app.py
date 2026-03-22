import os
import re
import json
import psycopg2
from flask import Flask, request, jsonify, redirect, send_from_directory
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder='frontend/dist', static_url_path='/')
CORS(app)

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("WARNING: GROQ_API_KEY not found in environment or .env file.")
MODEL = "llama-3.3-70b-versatile"

NARA_BASE_URL = "https://storage.googleapis.com/jfkweb-prod"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# Load prompts from files (falls back to inline if file not found)
# Check both: rag/prompts/ (Docker/local) and project-root/prompts/ (dev)
_prompts_local = os.path.join(os.path.dirname(__file__), 'prompts')
_prompts_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts')
PROMPTS_DIR = _prompts_local if os.path.isdir(_prompts_local) else _prompts_root


def load_prompt(filename, fallback=""):
    """Load a prompt from the prompts directory, preferring optimized version."""
    optimized_path = os.path.join(PROMPTS_DIR, 'optimized', filename)
    base_path = os.path.join(PROMPTS_DIR, filename)
    for path in [optimized_path, base_path]:
        if os.path.exists(path):
            with open(path) as f:
                return f.read()
    return fallback


def ensure_fts_index():
    """Create full-text search GIN index if it doesn't exist."""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_jfk_pages_content_fts
            ON jfk_pages USING GIN (to_tsvector('english', content))
        """)
        conn.commit()
        cur.close()
        conn.close()
        print("FTS index ready.")
    except Exception as e:
        print(f"FTS index creation skipped: {e}")


if DATABASE_URL:
    ensure_fts_index()


def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    query = data.get('query')
    history = data.get('history', [])

    if not query:
        return jsonify({"error": "No query provided"}), 400

    if not client:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Step 0: Check if query contains a document ID (e.g., 104-10433-10209)
        doc_id_match = re.search(r'\b(\d{3}-\d{5}-\d{5})\b', query)
        if doc_id_match:
            doc_id = doc_id_match.group(1)
            filename = f"{doc_id}.pdf"
            print(f"Document ID detected: {doc_id}")

            cur.execute(
                "SELECT content, filename, page_number FROM jfk_pages WHERE filename = %s ORDER BY page_number",
                (filename,)
            )
            doc_results = cur.fetchall()

            if doc_results:
                # Build context from all pages of this document
                context_parts = []
                for idx, r in enumerate(doc_results, 1):
                    context_parts.append(f"[{idx}] Source: {r[1]}, Page {r[2]}\n{r[0]}")
                context = "\n\n".join(context_parts)

                doc_instructions = load_prompt('document-agent.txt').replace('{filename}', filename)

                messages_list = [
                    {"role": "system", "content": doc_instructions},
                    {"role": "user", "content": f"DOCUMENT PAGES:\n{context}\n\nUSER INQUIRY: {query}"},
                ]

                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_list,
                    temperature=0.3,
                )

                return jsonify({
                    "answer": completion.choices[0].message.content,
                    "sources": [{"filename": r[1], "page": r[2]} for r in doc_results],
                    "query_type": "document"
                })
            else:
                return jsonify({
                    "answer": f"Document **{filename}** was not found in the archive. Please verify the document ID.",
                    "sources": [],
                    "query_type": "document"
                })

        # Step 1: Analyze query intent and extract keywords
        # Include recent conversation history so the router can resolve
        # follow-up references like "his", "that document", "tell me more"
        history_context = ""
        if history:
            recent = history[-6:]  # last 3 exchanges
            history_lines = []
            for msg in recent:
                role = msg.get('role', 'user')
                if role in ('user', 'assistant'):
                    # Truncate long assistant messages
                    content = msg['content'][:200] if role == 'assistant' else msg['content']
                    history_lines.append(f"{role}: {content}")
            if history_lines:
                history_context = "\n\nConversation history (for context):\n" + "\n".join(history_lines)

        analysis_prompt = load_prompt('router.txt').replace('{query}', query) + history_context

        try:
            analysis_res = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            analysis_data = json.loads(analysis_res.choices[0].message.content)
            search_terms = analysis_data.get('keywords', query.split())
            query_type = analysis_data.get('type', 'research')
            needs_retrieval = analysis_data.get('needs_retrieval', True)
            metadata_filter = analysis_data.get('metadata_filter', None)
        except Exception as e:
            print(f"Query keyword analysis failed: {e}")
            search_terms = query.split()
            query_type = 'research'
            needs_retrieval = True
            metadata_filter = None

        # Handle metadata queries (redactions, handwriting, stamps, tables, etc.)
        if query_type == "metadata" and metadata_filter:
            print(f"Query type: METADATA — {metadata_filter}")
            allowed_columns = {
                'has_redactions', 'includes_handwriting', 'has_stamps',
                'has_tables', 'has_forms', 'is_typewritten', 'document_type'
            }
            col = metadata_filter.get('column', '')
            val = metadata_filter.get('value', True)

            if col not in allowed_columns:
                # Fallback to regular search if column is invalid
                query_type = 'simple'
            else:
                # Get count
                if isinstance(val, bool):
                    cur.execute(f"SELECT COUNT(*) FROM jfk_pages WHERE {col} = %s", (val,))
                    total_matching = cur.fetchone()[0]
                    cur.execute(f"SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE {col} = %s", (val,))
                    docs_matching = cur.fetchone()[0]

                    # Get sample documents
                    cur.execute(f"""
                        SELECT DISTINCT ON (file_id) content, filename, page_number, file_id
                        FROM jfk_pages WHERE {col} = %s AND content IS NOT NULL
                        ORDER BY file_id, page_number
                        LIMIT 10
                    """, (val,))
                else:
                    cur.execute(f"SELECT COUNT(*) FROM jfk_pages WHERE {col} ILIKE %s", (f"%{val}%",))
                    total_matching = cur.fetchone()[0]
                    cur.execute(f"SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE {col} ILIKE %s", (f"%{val}%",))
                    docs_matching = cur.fetchone()[0]

                    cur.execute(f"""
                        SELECT DISTINCT ON (file_id) content, filename, page_number, file_id
                        FROM jfk_pages WHERE {col} ILIKE %s AND content IS NOT NULL
                        ORDER BY file_id, page_number
                        LIMIT 10
                    """, (f"%{val}%",))

                sample_results = cur.fetchall()

                # Build context with metadata stats + samples
                context_parts = [
                    f"METADATA QUERY RESULTS for {col} = {val}:",
                    f"Total matching pages: {total_matching}",
                    f"Total matching documents: {docs_matching}",
                    "",
                    "SAMPLE DOCUMENTS:"
                ]
                sources_list = []
                for idx, r in enumerate(sample_results, 1):
                    snippet = r[0][:500] if r[0] else "(no content)"
                    context_parts.append(f"[{idx}] Source: {r[1]}, Page {r[2]}\n{snippet}")
                    sources_list.append({"filename": r[1], "page": r[2]})
                context = "\n\n".join(context_parts)

                meta_prompt = f"""You are a research historian answering questions about the JFK document archive.
The user asked a question about document metadata. Below are the query results from the database.
Present the statistics clearly, then briefly describe the sample documents found.

CRITICAL FORMATTING RULES:
- NEVER show reasoning steps. No "Step 1:", no LaTeX, no "The final answer is".
- Write your answer directly.
- Cite sample documents as [1], [2], etc.

{context}

USER INQUIRY: {query}"""

                messages_list = [{"role": "user", "content": meta_prompt}]
                completion = client.chat.completions.create(
                    model=MODEL,
                    messages=messages_list,
                    temperature=0.3,
                )

                return jsonify({
                    "answer": completion.choices[0].message.content,
                    "sources": sources_list,
                    "query_type": "metadata"
                })

        # Handle conversational queries (greetings, thanks, etc.) without DB retrieval
        if not needs_retrieval or query_type == "conversational":
            print(f"Query type: CONVERSATIONAL — skipping retrieval")
            conv_messages = [{"role": "system", "content": load_prompt('conversational.txt')}]
            for msg in history[-20:]:
                role = msg.get('role', 'user')
                if role in ('user', 'assistant'):
                    conv_messages.append({"role": role, "content": msg['content']})
            conv_messages.append({"role": "user", "content": query})

            completion = client.chat.completions.create(
                model=MODEL,
                messages=conv_messages,
                temperature=0.5,
            )
            return jsonify({
                "answer": completion.choices[0].message.content,
                "sources": [],
                "query_type": "conversational"
            })

        # Filter out empty strings
        search_terms = [t for t in search_terms if t.strip()]
        if not search_terms:
            search_terms = [query]

        # Configure strategy based on query type
        if query_type == "simple":
            context_limit = 10
            print(f"Query type: SIMPLE (Terms: {search_terms})")
        else:
            context_limit = 20
            print(f"Query type: RESEARCH (Terms: {search_terms})")

        # Build search query using PostgreSQL full-text search for relevance ranking
        # Use plainto_tsquery which safely handles multi-word terms and natural language
        # Combine all search terms into one query string for plainto_tsquery
        ts_query_input = ' '.join(search_terms)

        search_query = """
            SELECT content, filename, page_number
            FROM (
                SELECT DISTINCT ON (left(content, 200)) content, filename, page_number,
                    ts_rank_cd(to_tsvector('english', content), plainto_tsquery('english', %s)) AS rank_score
                FROM jfk_pages
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
            ) sub
            ORDER BY rank_score DESC, length(content) DESC
            LIMIT 30
        """
        full_params = [ts_query_input, ts_query_input]

        # Step 2: Get global stats
        cur.execute("SELECT COUNT(*) FROM jfk_pages")
        total_p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE includes_handwriting = true")
        hw_p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_stamps = true")
        stamp_p = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_redactions = true")
        redact_p = cur.fetchone()[0]

        cur.execute(search_query, full_params)
        results = cur.fetchall()

        # Fallback to ILIKE if full-text search returns no results (handles OCR artifacts)
        if not results:
            print("Full-text search returned 0 results, falling back to ILIKE")
            where_clauses = [f"content ILIKE %s" for _ in search_terms]
            fallback_query = f"""
                SELECT content, filename, page_number
                FROM (
                    SELECT DISTINCT ON (left(content, 200)) content, filename, page_number
                    FROM jfk_pages
                    WHERE ({' OR '.join(where_clauses)})
                ) sub
                ORDER BY length(content) DESC
                LIMIT 30
            """
            fallback_params = [f"%{term}%" for term in search_terms]
            cur.execute(fallback_query, fallback_params)
            results = cur.fetchall()

        # Deduplicate and clean results
        unique_results = []
        seen_content = set()
        for r in results:
            content_snippet = r[0][:200].strip()
            if content_snippet not in seen_content:
                unique_results.append(r)
                seen_content.add(content_snippet)

        # Re-rank: ask LLM to pick the most relevant results for the query
        if len(unique_results) > context_limit:
            snippets_for_rerank = []
            for idx, r in enumerate(unique_results):
                # Send first 300 chars of each result to keep token usage low
                snippet = r[0][:300].replace('\n', ' ').strip()
                snippets_for_rerank.append(f"[{idx}] {r[1]}, Page {r[2]}: {snippet}")

            rerank_prompt = f"""You are a relevance judge. Given a user query and a list of document snippets, return the indices of the {context_limit} most relevant snippets as a JSON array of integers, ordered by relevance (most relevant first).

User query: "{query}"

Snippets:
{chr(10).join(snippets_for_rerank)}

Return ONLY a JSON array of integers, e.g. [3, 0, 7, 1, ...]"""

            try:
                rerank_res = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": rerank_prompt}],
                    temperature=0,
                    response_format={"type": "json_object"}
                )
                rerank_raw = json.loads(rerank_res.choices[0].message.content)
                # Handle both {"indices": [...]} and plain [...]
                if isinstance(rerank_raw, dict):
                    indices = list(rerank_raw.values())[0]
                else:
                    indices = rerank_raw
                # Filter valid indices and rebuild results in reranked order
                valid_indices = [i for i in indices if isinstance(i, int) and 0 <= i < len(unique_results)]
                final_results = [unique_results[i] for i in valid_indices[:context_limit]]
                print(f"Re-ranked: selected {len(final_results)} from {len(unique_results)} candidates")
            except Exception as e:
                print(f"Re-ranking failed, using FTS order: {e}")
                final_results = unique_results[:context_limit]
        else:
            final_results = unique_results[:context_limit]

        # Build numbered source context for in-text citations
        context = ""
        if final_results:
            context_parts = []
            for idx, r in enumerate(final_results, 1):
                context_parts.append(f"[{idx}] Source: {r[1]}, Page {r[2]}\n{r[0]}")
            context = "\n\n".join(context_parts)
        else:
            context = "NO SEARCH RESULTS FOUND."

        # Load prompt based on query type
        if query_type == "simple":
            instructions = load_prompt('rag-simple.txt')
        else:
            instructions = load_prompt('rag-research.txt')

        system_prompt = f"""{instructions}

CRITICAL FORMATTING RULES:
- NEVER show your reasoning process. No "Step 1:", "Step 2:", "Let me analyze", etc.
- NEVER use LaTeX, $\\boxed{{}}$, or math formatting.
- NEVER write "The final answer is".
- Write your answer directly as a research historian would present findings to a reader.

ARCHIVE METADATA:
- Total Archive: {total_p:,} pages across the collection
- Pages with Handwriting: {hw_p:,}
- Pages with Official Stamps: {stamp_p:,}
- Pages with Redactions: {redact_p:,}
"""

        user_prompt = f"""RETRIEVED DOCUMENTS:
{context}

USER INQUIRY: {query}

IMPORTANT REMINDERS:
- You MUST cite sources using [1], [2], etc. for EVERY factual claim. A response without any citations is a failure.
- ONLY use information from the RETRIEVED DOCUMENTS above. Do not use your own knowledge.
- No reasoning steps, no LaTeX, no "Step 1/2/3"."""

        # Build messages — no conversation history for the RAG agent
        # History is only used by the Router Agent to resolve follow-up queries.
        # Including it here causes the LLM to mix in data from previous answers.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )

        answer_text = completion.choices[0].message.content

        # Post-process: strip chain-of-thought artifacts that LLaMA sometimes produces
        # Remove LaTeX $\boxed{...}$ patterns
        answer_text = re.sub(r'\$\\boxed\{([^}]*)\}\$', r'\1', answer_text)
        # Remove "The final answer is: ..." lines
        answer_text = re.sub(r'(?m)^.*The final answer is:?.*$', '', answer_text)
        # Remove "Step N:" reasoning headers
        answer_text = re.sub(r'(?m)^#+?\s*Step \d+:.*$', '', answer_text)
        # Clean up excessive blank lines left behind
        answer_text = re.sub(r'\n{3,}', '\n\n', answer_text).strip()

        all_sources = [{"filename": r[1], "page": r[2]} for r in final_results]

        # Safety net: if LLM produced no citations but we have sources, retry once
        has_citations = bool(re.search(r'\[\d+\]', answer_text))
        if not has_citations and final_results:
            print("WARNING: No citations in answer, retrying with stricter prompt")
            retry_prompt = f"""The previous answer had NO citations. This is unacceptable.

RETRIEVED DOCUMENTS:
{context}

USER INQUIRY: {query}

You MUST rewrite your answer using ONLY the documents above. Every single factual sentence MUST end with a citation like [1], [2], etc. If the documents don't contain relevant info, say so. Do NOT use your own knowledge."""

            retry_messages = [{"role": "system", "content": system_prompt}]
            retry_messages.append({"role": "user", "content": retry_prompt})
            retry_completion = client.chat.completions.create(
                model=MODEL,
                messages=retry_messages,
                temperature=0.2,
            )
            retry_text = retry_completion.choices[0].message.content
            # Use retry if it has citations, otherwise keep original
            if re.search(r'\[\d+\]', retry_text):
                answer_text = retry_text
                # Re-apply post-processing
                answer_text = re.sub(r'\$\\boxed\{([^}]*)\}\$', r'\1', answer_text)
                answer_text = re.sub(r'(?m)^.*The final answer is:?.*$', '', answer_text)
                answer_text = re.sub(r'(?m)^#+?\s*Step \d+:.*$', '', answer_text)
                answer_text = re.sub(r'\n{3,}', '\n\n', answer_text).strip()

        # Remap citations to only include actually-cited sources
        # Find which [N] indices appear in the answer
        cited_nums = sorted(set(int(m) for m in re.findall(r'\[(\d+)\]', answer_text)))
        if cited_nums:
            # Build new sources list with only cited ones, and remap [old] -> [new] in text
            new_sources = []
            remap = {}
            for new_idx, old_num in enumerate(cited_nums, 1):
                old_idx = old_num - 1
                if 0 <= old_idx < len(all_sources):
                    new_sources.append(all_sources[old_idx])
                    remap[old_num] = new_idx

            # Replace citation numbers in answer text (largest first to avoid [1] replacing inside [10])
            for old_num in sorted(remap.keys(), reverse=True):
                answer_text = answer_text.replace(f'[{old_num}]', f'[__CITE_{remap[old_num]}__]')
            for new_num in remap.values():
                answer_text = answer_text.replace(f'[__CITE_{new_num}__]', f'[{new_num}]')

            sources_out = new_sources
        else:
            sources_out = all_sources

        return jsonify({
            "answer": answer_text,
            "sources": sources_out,
            "query_type": query_type
        })
    except Exception as e:
        print(f"Error in /api/chat: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route('/api/stats', methods=['GET'])
def stats():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM jfk_pages")
        total_pages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages")
        total_docs = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0")
        pages_with_content = cur.fetchone()[0]
        cur.execute("SELECT COUNT(DISTINCT file_id) FROM jfk_pages WHERE content IS NOT NULL AND length(trim(content)) > 0")
        docs_with_content = cur.fetchone()[0]

        page_pct = (pages_with_content / total_pages * 100) if total_pages > 0 else 0
        doc_pct = (docs_with_content / total_docs * 100) if total_docs > 0 else 0

        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE includes_handwriting = true")
        handwritten_pages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_stamps = true")
        stamped_pages = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM jfk_pages WHERE has_redactions = true")
        redacted_pages = cur.fetchone()[0]

        cur.execute("SELECT document_type, COUNT(*) as count FROM jfk_pages GROUP BY document_type ORDER BY count DESC LIMIT 5")
        doc_types = cur.fetchall()

        return jsonify({
            "total_pages": total_pages,
            "total_docs": total_docs,
            "pages_with_content": pages_with_content,
            "docs_with_content": docs_with_content,
            "page_content_pct": round(page_pct, 1),
            "doc_content_pct": round(doc_pct, 1),
            "handwritten_pages": handwritten_pages,
            "stamped_pages": stamped_pages,
            "redacted_pages": redacted_pages,
            "document_types": [{"type": r[0], "count": r[1]} for r in doc_types]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        cur.close()
        conn.close()


@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    action = data.get('action')
    text = data.get('text')

    if not text:
        return jsonify({"error": "No text provided"}), 400

    if not client:
        return jsonify({"error": "GROQ_API_KEY not configured"}), 500

    if action == 'names':
        prompt = f"Extract all unique people's names from the following text. Return them as a comma-separated list. If none, say 'None'.\n\nText: {text}"
    elif action == 'summarize':
        prompt = f"Provide a concise summary of the following text, highlighting key information and events.\n\nText: {text}"
    else:
        return jsonify({"error": "Invalid action"}), 400

    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return jsonify({"result": completion.choices[0].message.content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/pdf/<filename>', methods=['GET'])
def get_pdf(filename):
    """Redirect to NARA archive URL for the PDF."""
    # Strip .pdf extension to get the file_id, then build the NARA URL
    file_id = filename.replace('.pdf', '').replace('.PDF', '')
    nara_url = f"{NARA_BASE_URL}/{file_id}.pdf"
    return redirect(nara_url)


# SPA catch-all: serve frontend for non-API routes
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')


@app.errorhandler(404)
def not_found(e):
    # For SPA routing: serve index.html for any unmatched route
    if not request.path.startswith('/api'):
        return send_from_directory(app.static_folder, 'index.html')
    return jsonify({"error": "Not found"}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
