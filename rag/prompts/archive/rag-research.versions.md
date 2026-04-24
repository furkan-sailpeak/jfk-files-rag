# Prompt version history — `rag-research`

- **v1** = early thorough baseline (pre-simplification), archived in `rag-research.v1.txt`.
- **v2** = current short live baseline, archived in `rag-research.v2.txt`; identical to `rag-research.txt`.
- **v3+** = optimizer outputs.

The live prompt read by app.py is whichever version sits in `rag/prompts/optimized/rag-research.txt`; if that file is absent, app.py falls back to `rag/prompts/rag-research.txt`.

| Version | Timestamp | Seed fitness | Final fitness | Δ | Iters | Notes |
|---|---|---|---|---|---|---|
| v1 | 2026-04-24T22:07:44 | 0.547 | 0.592 | +0.045 | 6 | The revision adds tighter instructions to distinguish direct evidence from speculation, tailor output to document-list a · The revision tightens rules for partial evidence, timelines, relationship claims, and document-identification answers so |
| v1 | 2026-04-24T23:36:00 | 0.619 | 0.684 | +0.000 | 150 | GEPA run_dir=/Users/furkandemir/Desktop/thesis-rag-dashboard/eval/gepa_run_rag-research |
