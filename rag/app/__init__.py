"""JFK Files RAG backend. Request flow: main (routes) -> pipeline (one function
per answer path) -> retrieval / generation / verification -> postprocess."""
