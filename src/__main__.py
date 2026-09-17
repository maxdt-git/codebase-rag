from pathlib import Path
import fire
from src.indexer import build_index


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


class RagCLI:
    def index(
            self,
            max_chunk_size: int = 2000
    ) -> None:
        build_index(RAW_DIR, PROCESSED_DIR, max_chunk_size)


if __name__ == "__main__":
    fire.Fire(RagCLI)
