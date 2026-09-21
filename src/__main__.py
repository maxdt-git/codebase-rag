from pathlib import Path
import fire
from src.indexer import build_index
from src.retriever import load_retriever, search_data, search as search_single
import json
from src.models import StudentSearchResults, RagDataset
from src.evaluate import build_ground_truth, compute_recall
from src.generator import load_model, generate_answer
from src.generator import answer_dataset as generate_dataset_answers


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"


class RagCLI:
    """
    Command-line interface of the RAG pipeline
    """
    def index(
            self,
            max_chunk_size: int = 2000,
            raw_dir: str = RAW_DIR,
            processed_dir: str = PROCESSED_DIR
    ) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/
        """
        if not isinstance(max_chunk_size, int) \
                or not 1 <= max_chunk_size <= 2000:
            raise ValueError("max_chunk_size must be between 1 and 2000")
        build_index(Path(raw_dir), Path(processed_dir), max_chunk_size)

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results",
            processed_dir: str = PROCESSED_DIR
    ) -> None:
        """
        Run search over a dataset and save a StudentSearchResults JSON
        """
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non negative integer")
        retriever = load_retriever(Path(processed_dir))
        results = search_data(Path(dataset_path), retriever, k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        # keep the same file name as the input dataset, just a new folder
        output_path = save_dir / Path(dataset_path).name
        output_path.write_text(
            results.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results in {output_path}")

    def evaluate(
            self,
            student_search_results_path: str,
            dataset_path: str
    ) -> None:
        """
        Report recall@k of a StudentSearchResults against a dataset
        """
        with open(student_search_results_path, encoding="utf-8") as f:
            student_results = StudentSearchResults.model_validate(json.load(f))

        with open(dataset_path, encoding="utf-8") as f:
            dataset = RagDataset.model_validate(json.load(f))

        ground_truth = build_ground_truth(dataset)
        recall = compute_recall(student_results, ground_truth)
        print(f"recall@k = {recall:.3f}")

    def search(
            self,
            query: str,
            k: int = 10,
            processed_dir: str = PROCESSED_DIR
    ) -> None:
        """
        Return the top-k sources for a query
        """
        query = str(query)
        if not isinstance(k, int) or k < 0:
            raise ValueError("k must be a non negative integer")

        retriever = load_retriever(Path(processed_dir))
        results = search_single(query, retriever, k)
        if not results:
            print("No result found")
        for source in results:
            print(source)

    def answer(
            self,
            query: str,
            k: int = 10,
            processed_dir: str = PROCESSED_DIR
    ) -> None:
        """
        Answer a single query with retreive
        """
        query = str(query)
        retriever = load_retriever(Path(processed_dir))
        k = min(k, retriever.scores["num_docs"])
        sources = search_single(query, retriever, k)
        tokenizer, model = load_model()
        print(generate_answer(query, sources, tokenizer, model))

    def answer_dataset(
            self,
            student_search_results_path: str,
            save_directory: str
    ) -> None:
        """
        Generate answers for a dataset and save the result in json
        """
        with open(student_search_results_path, encoding="utf-8") as f:
            student_results = StudentSearchResults.model_validate(json.load(f))

        tokenizer, model = load_model()
        # reuse the sources search_dataset already found, no re-search here
        result = generate_dataset_answers(student_results, tokenizer, model)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        # keep the same file name as the input file, just a new folder
        output_path = save_dir / Path(student_search_results_path).name
        output_path.write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results_and_answer to {output_path}")


if __name__ == "__main__":
    try:
        fire.Fire(RagCLI)
    except Exception as e:
        print(f"Error: {e}")
