from pathlib import Path
import bm25s
from src.models import MinimalSource


def load_retriever(processed_dir: Path) -> bm25s.BM25:
    """
    Load the BM25 index saved under processed_dir
    """
    return bm25s.BM25.load(str(processed_dir), load_corpus=True)


def search(
        query: str,
        retriever: bm25s.BM25,
        k: int
) -> list[MinimalSource]:
    """
    Retrieve the top-k sources for a single query
    """
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    documents, scores = retriever.retrieve(
        query_tokens, k=k, show_progress=False)

    return [MinimalSource(**doc) for doc in documents[0]]


# if __name__ == "__main__":
#     retriever = load_retriever(Path("data/processed"))
#     results = search("How to configure the OpenAI server?", retriever, k=5)
#     for source in results:
#         print(source)
