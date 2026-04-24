# Prompt version history — `rag-simple`

- **v1** = early thorough baseline (pre-simplification), archived in `rag-simple.v1.txt`.
- **v2** = current short live baseline, archived in `rag-simple.v2.txt`; identical to `rag-simple.txt`.
- **v3+** = optimizer outputs.

The live prompt read by app.py is whichever version sits in `rag/prompts/optimized/rag-simple.txt`; if that file is absent, app.py falls back to `rag/prompts/rag-simple.txt`.

| Version | Timestamp | Seed fitness | Final fitness | Δ | Iters | Notes |
|---|---|---|---|---|---|---|
| v1 | 2026-04-24T23:16:41 | 0.759 | 0.797 | +0.000 | 150 | GEPA run_dir=/Users/furkandemir/Desktop/thesis-rag-dashboard/eval/gepa_run_rag-simple |
