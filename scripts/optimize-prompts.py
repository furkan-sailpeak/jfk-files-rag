"""
Prompt Optimization Pipeline for JFK RAG System.
Uses leo-prompt-optimizer to optimize the prompts in /prompts/ folder.

Usage:
  pip install leo-prompt-optimizer python-dotenv
  python scripts/optimize-prompts.py

Requires GROQ_API_KEY in .env (already used by the main app).
Outputs optimized prompts to prompts/optimized/ for review.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

PROMPTS_DIR = PROJECT_ROOT / "prompts"
OUTPUT_DIR = PROMPTS_DIR / "optimized"

PROMPTS = [
    {
        "file": "router.txt",
        "description": "Router agent that classifies user queries for a JFK document RAG system into types and extracts search keywords",
        "user_input_example": "What connections did Oswald have with the Soviet embassy in Mexico City?",
        "llm_output_example": '{"keywords": ["Oswald", "Soviet embassy", "Mexico City"], "type": "research", "needs_retrieval": true}',
    },
    {
        "file": "document-agent.txt",
        "description": "System prompt for answering questions about a single specific declassified JFK document with page-level citations",
        "user_input_example": "What is this document about? DOCUMENT PAGES:\n[1] Source: 104-10433-10209.pdf, Page 1\nMEMORANDUM FOR: Director of Central Intelligence...",
        "llm_output_example": (
            "## Document Overview\n\nThis is a **CIA memorandum** addressed to the Director of Central Intelligence [1]. "
            "The document discusses surveillance activities related to **Lee Harvey Oswald's** movements [1]."
        ),
    },
    {
        "file": "conversational.txt",
        "description": "System prompt for handling greetings, off-topic, and system questions without database retrieval",
        "user_input_example": "Hello, what can you do?",
        "llm_output_example": (
            "Hello! I'm the JFK Files Research System. I can help you search through declassified JFK assassination documents. "
            "You can ask me about people, events, organizations, or even search for specific document IDs like 104-10433-10209."
        ),
    },
    {
        "file": "rag-simple.txt",
        "description": "System prompt for generating direct answers with citations from retrieved JFK archival documents",
        "user_input_example": "RETRIEVED DOCUMENTS:\n[1] Source: 104-10433-10209.pdf, Page 5\nOswald was observed entering the Soviet Embassy...\n\nUSER INQUIRY: Did Oswald visit the Soviet Embassy?",
        "llm_output_example": (
            "## Soviet Embassy Visit\n\nYes, **Lee Harvey Oswald** was observed entering the Soviet Embassy [1]. "
            "The document confirms surveillance recorded his visit to the embassy grounds [1]."
        ),
    },
    {
        "file": "rag-research.txt",
        "description": "System prompt for generating structured research reports with citations from retrieved JFK archival documents",
        "user_input_example": "RETRIEVED DOCUMENTS:\n[1] Source: 104-10433-10209.pdf, Page 5\nOswald was observed...\n[2] Source: 104-10434-10210.pdf, Page 12\nCIA station reported...\n\nUSER INQUIRY: Analyze Oswald's connections to intelligence agencies",
        "llm_output_example": (
            "## Executive Summary\n\nThe retrieved documents reveal surveillance of **Oswald** by CIA operatives [1] "
            "and communication between CIA stations regarding his activities [2].\n\n"
            "## Detailed Findings\n\n### CIA Surveillance\nOswald was observed entering the Soviet Embassy [1]..."
        ),
    },
]


def main():
    try:
        from leo_prompt_optimizer import GroqProvider, LeoOptimizer
    except ImportError:
        print("ERROR: leo-prompt-optimizer not installed.")
        print("Run: pip install leo-prompt-optimizer")
        sys.exit(1)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not found in .env")
        sys.exit(1)

    provider = GroqProvider()
    optimizer = LeoOptimizer(provider, default_model="llama-3.3-70b-versatile")

    OUTPUT_DIR.mkdir(exist_ok=True)

    for prompt_def in PROMPTS:
        src = PROMPTS_DIR / prompt_def["file"]
        if not src.exists():
            print(f"SKIP: {prompt_def['file']} not found")
            continue

        original = src.read_text()
        print(f"\n{'='*60}")
        print(f"Optimizing: {prompt_def['file']}")
        print(f"{'='*60}")

        try:
            optimized = optimizer.optimize(
                prompt_draft=original,
                user_input_example=prompt_def.get("user_input_example", ""),
                llm_output_example=prompt_def.get("llm_output_example", ""),
                top_instruction=prompt_def.get("description", ""),
                model="llama-3.3-70b-versatile",
            )

            out_path = OUTPUT_DIR / prompt_def["file"]
            out_path.write_text(optimized)
            print(f"  -> Saved to {out_path.relative_to(PROJECT_ROOT)}")
            print(f"  Original: {len(original)} chars")
            print(f"  Optimized: {len(optimized)} chars")

        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Done! Review optimized prompts in: prompts/optimized/")
    print("If happy, copy them back to prompts/ and update app.py to load from files.")


if __name__ == "__main__":
    main()
