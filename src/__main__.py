from pathlib import Path
import fire
from src.indexer import build_index
from src.retriever import load_retriever, search_data


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class RagCLI:
    def index(
            self,
            max_chunk_size: int = 2000
    ) -> None:
        build_index(RAW_DIR, PROCESSED_DIR, max_chunk_size)

    def search_dataset(
            self,
            dataset_path: str,
            k: int = 10,
            save_directory: str = "data/output/search_results"
    ) -> None:
        retriever = load_retriever(PROCESSED_DIR)
        results = search_data(Path(dataset_path), retriever, k)

        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / Path(dataset_path).name
        output_path.write_text(
            results.model_dump_json(indent=2), encoding="utf-8")
        print(f"Saved student_search_results in {output_path}")


if __name__ == "__main__":
    fire.Fire(RagCLI)
