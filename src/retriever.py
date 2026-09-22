from pathlib import Path
import bm25s
from src.models import MinimalSource, StudentSearchResults
from src.models import RagDataset, MinimalSearchResults
import json
from tqdm import tqdm


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
    # tokenize the question the same way the chunks were tokenized
    k = min(k, retriever.scores["num_docs"])
    query_tokens = bm25s.tokenize([query], stopwords="en", show_progress=False)
    documents, scores = retriever.retrieve(
        query_tokens, k=k, show_progress=False)
    # documents[0]: results for our single query, already MinimalSource dicts
    return [MinimalSource(**doc) for doc, score in zip(
        documents[0], scores[0]) if score > 0]


def search_data(
        dataset_path: Path,
        retriever: bm25s.BM25,
        k: int
) -> StudentSearchResults:
    """
    Run search over every question of a dataset
    """
    with open(dataset_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    dataset = RagDataset.model_validate(raw_data)

    minimal_results: list[MinimalSearchResults] = []
    # search() once per question, question_id/question carried over as-is
    for question in tqdm(dataset.rag_questions, desc="Searching"):
        sources = search(question.question, retriever, k)
        minimal_results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )
    return StudentSearchResults(search_results=minimal_results, k=k)


# if __name__ == "__main__":
#     retriever = load_retriever(Path("data/processed"))
#     results = search("How to configure the OpenAI server?", retriever, k=5)
#     for source in results:
#         print(source)
#     dataset_path = Path(
#         "data/datasets/UnansweredQuestions/dataset_docs_public.json")
#     test = search_data(dataset_path, retriever, k=5)
#     print(test.search_results[0])
