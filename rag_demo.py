from rich.panel import Panel

from ragstack.console import console
from rag_core import MarkdownRag

def main():
    rag = MarkdownRag()
    query = "What are the main concepts described in these markdown documents?"
    context = rag.build_context(query, k=5, max_chars=3000)

    console.print(Panel(context, title="RAG Context", style="info"))

if __name__ == "__main__":
    main()
