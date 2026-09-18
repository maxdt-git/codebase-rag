from pathlib import Path
import fire
from src.indexer import build_index
from src.retriever import load_retriever, search_data
import json
from src.models import StudentSearchResults, RagDataset
from src.evaluate import build_ground_truth, compute_recall


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class RagCLI:
    def index(
            self,
            max_chunk_size: int = 2000
    ) -> None:
        """
        Ingest data/raw/ and build the index under data/processed/
        """
        build_index(RAW_DIR, PROCESSED_DIR, max_chunk_size)

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results"
    ) -> None:
        """
        Run search over a dataset and save a StudentSearchResults JSON
        """
        retriever = load_retriever(PROCESSED_DIR)
        results = search_data(Path(dataset_path), retriever, k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    fire.Fire(RagCLI)
