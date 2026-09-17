from pathlib import Path
from src.chunking.markdown_chunker import chunk_markdown
from src.chunking.python_chunker import chunk_python
from tqdm import tqdm
import bm25s


def go_through_files(raw_dir: Path) -> list[Path]:
    """
    Go through the directory and get all .py, .txt. md
    """
    all_files_path: list[Path] = []
    for file in raw_dir.rglob('*'):
        if file.suffix == ".py" or file.suffix == ".md" \
           or file.suffix == ".txt":
            if file.is_file():
                all_files_path.append(file)

    return sorted(all_files_path)


def build_corpus(
        files: list[Path],
        max_chunk_size: int
) -> tuple[list[str], list[dict[str, str | int]]]:
    """
    Chunk every file and return the chunk texts with their source metadata
    """
    texts: list[str] = []
    metas: list[dict[str, str | int]] = []

    for file in tqdm(files, desc="Chunking"):
        try:
            content = file.read_text(encoding="utf-8")
        except Exception as e:
            print(e)
            continue
        if file.suffix == ".py":
            chunks = chunk_python(content, max_chunk_size)
        elif file.suffix == ".md" or file.suffix == ".txt":
            chunks = chunk_markdown(content, max_chunk_size)
        else:
            continue

        for start, end in chunks:
            texts.append(content[start:end])
            metas.append({"file_path": str(file),
                          "first_character_index": start,
                          "last_character_index": end})

    return texts, metas


def build_index(
        raw_dir: Path,
        processed_dir: Path,
        max_chunk_size: int
) -> None:
    """
    Build the BM25 index from raw_dir and save it under processed_dir
    """
    files = go_through_files(raw_dir)
    texts, metas = build_corpus(files, max_chunk_size)

    corpus_tokens = bm25s.tokenize(texts, stopwords="en")
    retriever = bm25s.BM25(k1=1.2, b=0.5, corpus=metas)
    retriever.index(corpus_tokens)

    processed_dir.mkdir(parents=True, exist_ok=True)
    retriever.save(str(processed_dir), corpus=metas)


# if __name__ == "__main__":
#     test = go_through_files(Path("data/raw/vllm-0.10.1"))
#     count_md_txt = 0
#     count_py = 0
#     for file in test:
#         if file.suffix == ".md" or file.suffix == ".txt":
#             count_md_txt += 1
#         if file.suffix == ".py":
#             count_py += 1
#     print("nb md + txt files: ", count_md_txt)
#     print("nb py files: ", count_py)
#     print("total files: ", count_md_txt + count_py)

#     files = go_through_files(Path('data/raw/vllm-0.10.1'))
#     texts, metas = build_corpus(files, max_chunk_size=2000)

#     print('nb chunks:', len(texts), len(metas))
#     print('coherence texts/metas:', len(texts) == len(metas))
#     print('exemple meta[0]:', metas[0])
#     print('exemple text[0] (30 premiers caracteres):', repr(texts[0][:30]))

#     print("=====")
#     build_index(Path("data/raw/vllm-0.10.1"), Path("data/processed"), 2000)
