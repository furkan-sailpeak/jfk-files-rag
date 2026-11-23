"""RAG backend (corpus set by CORPUS, see corpora/). Request flow: main (routes) -> pipeline (one function
per answer path) -> retrieval / generation / verification -> postprocess."""
