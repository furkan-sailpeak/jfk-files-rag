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

                doc_instructions = f"""You are a senior Research Historian. The user is asking about a specific declassified document: {filename}.
                All pages of this document have been retrieved below. Answer the user's question based SOLELY on this document's content.

                STRICT SOURCE RULES:
                - You may ONLY state facts explicitly written in this document.
                - Do NOT use outside knowledge.
                - If the document doesn't contain the requested information, say so.

                IN-TEXT CITATION RULES:
                - Cite page numbers using bracket notation: [1], [2], etc.
                - The numbers correspond to the page entries below.
                - Every factual sentence must have a citation.

                FORMATTING:
                - Use markdown: headers (##), **bold**, bullet points.
                - Start with a brief overview of the document, then address the user's specific question.
                """

                messages_list = [{"role": "system", "content": doc_instructions}]
                for msg in history[-20:]:
                    role = msg.get('role', 'user')
                    if role in ('user', 'assistant'):
                        messages_list.append({"role": role, "content": msg['content']})
                messages_list.append({"role": "user", "content": f"DOCUMENT PAGES:\n{context}\n\nUSER INQUIRY: {query}"})

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
        analysis_prompt = f"""Analyze this user query for a RAG system searching JFK files: '{query}'

        Return a valid JSON object with two keys:
        1. "keywords": A list of 1-3 most important search terms to find relevant documents.
        2. "type": "simple" (if asking for a specific name, date, fact, definition, or single document verification) or "research" (if asking for analysis, summary, relationships, or details on a broad topic).

        Reply ONLY with the JSON."""

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
        except Exception as e:
            print(f"Query keyword analysis failed: {e}")
            search_terms = query.split()
            query_type = 'research'

        # Configure strategy based on query type
        if query_type == "simple":
            context_limit = 10
            print(f"Query type: SIMPLE (Terms: {search_terms})")
        else:
            context_limit = 20
            print(f"Query type: RESEARCH (Terms: {search_terms})")

        # Build search query with PostgreSQL parameterized queries
        where_clauses = [f"content ILIKE %s" for _ in search_terms]
        ranking_clauses = [f"(CASE WHEN content ILIKE %s THEN 1 ELSE 0 END)" for _ in search_terms]
        rank_expr = f"({' + '.join(ranking_clauses)})"

        search_query = f"""
            SELECT content, filename, page_number
            FROM (
                SELECT DISTINCT ON (left(content, 200)) content, filename, page_number,
                    {rank_expr} AS rank_score
                FROM jfk_pages
                WHERE ({' OR '.join(where_clauses)})
            ) sub
            ORDER BY rank_score DESC, length(content) DESC
            LIMIT 30
        """
        search_params = [f"%{term}%" for term in search_terms]
        ranking_params = [f"%{term}%" for term in search_terms]
        full_params = ranking_params + search_params

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

        # Deduplicate and clean results
        unique_results = []
        seen_content = set()
        for r in results:
            content_snippet = r[0][:200].strip()
            if content_snippet not in seen_content:
                unique_results.append(r)
                seen_content.add(content_snippet)

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

        # Construct dynamic prompt instructions
        strict_rules = """
        STRICT SOURCE RULES:
        - You may ONLY state facts that are explicitly written in the RETRIEVED DOCUMENTS below.
        - Do NOT use any outside knowledge, prior training data, or general knowledge.
        - If the documents do not contain enough information to answer, say: "The retrieved documents do not contain sufficient information to answer this query."
        - Do NOT guess, infer beyond what is written, or fill gaps with external knowledge.
        - If a document is ambiguous, note the ambiguity rather than interpreting it.

        IN-TEXT CITATION RULES (MANDATORY):
        - Every claim, fact, or detail MUST have an in-text citation: [1], [2], etc.
        - The numbers correspond to the numbered sources in the RETRIEVED DOCUMENTS section.
        - Place citations at the end of the sentence they support: "Oswald traveled to Mexico City [3]."
        - If a fact appears in multiple sources, cite all: [1][4].
        - NEVER write a factual sentence without a citation.
        - NEVER cite a source for information that does not appear in that source's text.
        """

        if query_type == "simple":
            instructions = f"""You are a senior Research Historian. You answer questions about JFK assassination files based SOLELY on retrieved archival documents.

            FORMATTING:
            - Use markdown: headers (##), **bold** for key names/dates/places, and bullet points.
            - Start with a direct answer, then elaborate with supporting evidence from the documents.
            - Synthesize information across multiple sources when relevant.

            {strict_rules}"""
        else:
            instructions = f"""You are a senior Research Historian. You produce structured research reports about JFK assassination files based SOLELY on retrieved archival documents.

            REPORT STRUCTURE:
            1. **Executive Summary** — 2-3 sentence overview of key findings from the documents.
            2. **Detailed Findings** — Organized into logical sections with descriptive headers.
            3. **Cross-References** — Connections or contradictions between documents.
            4. **Archival Notes** — Mention any redactions, handwriting, or stamps noted in metadata.

            FORMATTING:
            - Use markdown: headers (##), **bold**, bullet points, and tables where appropriate.
            - Be comprehensive — extract every relevant detail from the documents.
            - Do NOT repeat the same fact in multiple sections.

            {strict_rules}"""

        system_prompt = f"""{instructions}

        ARCHIVE METADATA:
        - Total Archive: {total_p:,} pages across the collection
        - Pages with Handwriting: {hw_p:,}
        - Pages with Official Stamps: {stamp_p:,}
        - Pages with Redactions: {redact_p:,}
        """

        user_prompt = f"""RETRIEVED DOCUMENTS:
        {context}

        USER INQUIRY: {query}"""

        # Build messages with conversation history
        messages = [{"role": "system", "content": system_prompt}]

        # Include recent history (last 10 exchanges to stay within token limits)
        for msg in history[-20:]:
            role = msg.get('role', 'user')
            if role in ('user', 'assistant'):
                messages.append({"role": role, "content": msg['content']})

        messages.append({"role": "user", "content": user_prompt})

        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
        )

        return jsonify({
            "answer": completion.choices[0].message.content,
            "sources": [{"filename": r[1], "page": r[2]} for r in final_results],
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
