# Codebase RAG

**Stack:** Python · BM25 (`bm25s`) · Hugging Face Transformers / PyTorch · Qwen3-0.6B · Pydantic · uv

**Highlights**
- End-to-end RAG pipeline over a codebase: chunking → BM25 retrieval → grounded LLM answers
- **Recall@5: 0.86 on docs, 0.64 on code** (targets: 0.80 / 0.50)
- Indexes the full corpus in ~6 s; retrieves 199 questions in under 1 s

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a codebase.

The pipeline has four stages:

1. **Indexing**: every `.py`, `.md` and `.txt` file is split into chunks (at most 2000 characters) and indexed with BM25.
2. **Retrieving**: a question is matched against the index and the top-k source locations (`file_path` + character range) are returned.
3. **Augmenting**: the text of those sources is placed in the prompt.
4. **Generating**: `Qwen/Qwen3-0.6B` writes an answer grounded in that context.

Retrieval quality is measured with recall@k (a source is found when a retrieved chunk is in the same file and overlaps it with an IoU > 0.05).

## Instructions

**Requirements**: Python 3.10+, [uv](https://docs.astral.sh/uv/), about 6 GB of free disk space (PyTorch and the model weights) and an internet connection on the first run (the Qwen3-0.6B weights are downloaded from Hugging Face).

**Data** (not versioned, provided with the subject) must be laid out as follows:

```
data/raw/vllm-0.10.1/                       the corpus to index
data/datasets/AnsweredQuestions/*.json      ground-truth questions
data/datasets/UnansweredQuestions/*.json    questions only
```

`data/processed/` (the index) and `data/output/` (results) are generated.

**Install and run**:

```bash
make install     # uv sync
make index       # build the index (about 5 seconds)
make lint        # flake8 + mypy
make clean       # remove caches
make fclean      # also remove .venv, data/processed and data/output
```

`make run` + commad. Shortcuts exist for the whole pipeline (`search`, `search-dataset`, `evaluate`, `answer`, `answer-dataset`); variables `K`, `MAX_CHUNK_SIZE`, `DATASET` and `Q` can be change, e.g. `make evaluate K=5 DATASET=dataset_code_public.json`.

**CLI** (built with Python Fire, `uv run python -m src <command>`):

| Command | Options | Effect |
|---|---|---|
| `index` | `--max_chunk_size` (1 to 2000, default 2000), `--raw_dir` (default `data/raw`), `--processed_dir` (default `data/processed`) | Index `raw_dir` into `processed_dir` |
| `search <query>` | `--k`, `--processed_dir` | Print the top-k sources of one query |
| `search_dataset` | `--dataset_path --k --save_directory --processed_dir` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer <query>` | `--k`, `--processed_dir` | Search, then answer one query |
| `answer_dataset` | `--student_search_results_path --save_directory` | Answer every question of a search-results file, write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate` | `--student_search_results_path --dataset_path` | Print recall@k against a ground-truth dataset |

`--raw_dir` should stay a relative path: it is embedded verbatim in every `file_path` written to the index, and the moulinette compares that path as text.

Invalid input (empty query, `k=0`, invalid `k` or `max_chunk_size`, missing file, malformed JSON) prints an `Error: ...` message or `No result found` instead of a traceback.

## Example usage

```bash
uv run python -m src index --max_chunk_size 2000

uv run python -m src search "How to configure the OpenAI server?" --k 5

uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 5 --save_directory data/output/search_results/UnansweredQuestions

uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

## System architecture

```
data/raw ──► chunkers ──► indexer ──► data/processed (BM25 index)
                                            │
question ──► retriever (BM25) ◄─────────────┘
                 │
                 ├──► search_dataset ──► data/output/search_results ──► evaluate (recall@k)
                 │
                 └──► generator (Qwen3-0.6B) ──► answer_dataset ──► data/output/search_results_and_answer
```

Each command is a separate program run: they only communicate through files on disk. The index is built once, then reused by every search.

| File | Role |
|---|---|
| `src/__main__.py` | Fire CLI |
| `src/models.py` | pydantic data models (from the subject) |
| `src/chunking/python_chunker.py` | AST-based chunker for Python |
| `src/chunking/markdown_chunker.py` | Section/paragraph chunker for Markdown and text |
| `src/indexer.py` | File discovery, corpus building, BM25 index creation |
| `src/retriever.py` | Index loading, single and dataset search |
| `src/generator.py` | Prompt building, Qwen3 loading, answer generation |
| `src/evaluate.py` | IoU and recall@k |

## Chunking strategy

Both chunkers (`chunk_python`, `chunk_markdown`) return a list of `(start, end, context)` triples for a file's content. `start`/`end` are a character range `[start, end)` of the original file, never longer than `--max_chunk_size`; this is the only part saved to the index, as the `file_path` + character range of a `MinimalSource`. `context` is extra text prepended only to the copy of the chunk that gets indexed and tokenized for BM25 — it is never shown to the model and never changes `start`/`end`. Two strategies produce these triples, chosen by file extension.

**Python (`.py`)**: the file is parsed with `ast`, and each top-level function or class is one chunk, so a signature stays with its docstring and body (`context` is empty). Line/column positions from the AST are converted to character offsets. If a unit is too big:
- a class is split into its methods (recursively), each carrying the class name as `context`, so a method chunk still matches a question naming its class;
- a function or method that is still too big is cut into consecutive slices of `max_chunk_size` characters (no overlap), `context` carried over unchanged.

**Markdown and text (`.md`, `.txt`)**: the file is split at title lines (`#` to `######`). A section that is too big is split into paragraphs (blank-line separated), each one after the first carrying its section's title as `context` (the first paragraph already starts with the title text itself). A paragraph that is still too big is cut into raw slices. A file without titles, such as `CMakeLists.txt`, is one section.

On top of this, `indexer.py` prepends one more piece of `context` to every chunk regardless of type: the words of the file name (`lora_resolver.py` → `lora resolver`).

Result: 1969 files give 21 989 chunks, none longer than 2000 characters.

## Retrieval method

- **Algorithm**: BM25 (`bm25s` library), one "document" per chunk. Chunk and query texts are lowercased, split into words and stripped of English stop words.
- **Ranking**: `retrieve` scores every chunk and returns the top-k. Chunks with a score of 0 (no word in common with the question) are dropped, so an empty or nonsensical query returns nothing instead of random chunks. `k` is capped to the number of chunks.
- **Parameters**: `k1=1.5` (library default) and `b=0.5` (library default is 0.75), tuned on the public datasets: lowering `b` weakens the penalty BM25 applies to long chunks, which favors the large chunks produced by the AST chunker.
- **Persistence**: the index and one metadata line per chunk (`file_path` and character range) are saved in `data/processed/`. Chunk texts are not stored; for generation, the text is re-read from the corpus with the character range.
- **Generation**: the retrieved chunks are concatenated as context, followed by the question and an instruction to answer only from the context. `Qwen/Qwen3-0.6B` runs with its thinking mode disabled and greedy decoding (deterministic), 300 new tokens at most.

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions), and confirmed by the official moulinette.

| Recall | @1 | @3 | @5 | @10 | Required @5 |
|---|---|---|---|---|---|
| docs | 0.570 | 0.780 | **0.860** | 0.860 | 0.80 |
| code | 0.434 | 0.616 | **0.636** | 0.636 | 0.50 |

**Effect of `max_chunk_size`** (recall@5, `k1=1.5`, `b=0.5`):

| max_chunk_size | chunks | docs | code |
|---|---|---|---|
| **2000** | 21 989 | **0.860** | **0.636** |
| 1500 | 24 990 | 0.820 | 0.646 |
| 1000 | 30 620 | 0.820 | 0.616 |
| 500 | 46 913 | 0.770 | 0.606 |

Smaller chunks lower docs recall in every case: the chunks split from a large section lose the context that made it match the question. Code recall is flatter, since the filename and class-name context now carried in every chunk still matches regardless of chunk size.

**System performance**:

| | Measured | Limit |
|---|---|---|
| Indexing of the whole corpus | about 5-7 s | 5 min |
| Retrieval, 199 questions | under 1 s | 90 s |
| Answer generation (CPU) | about 22 s per question | none in the subject |

## Design decisions

- **BM25 instead of TF-IDF**: BM25 saturates the effect of repeated words and normalizes by chunk length, which suits chunks of very different sizes.
- **`bm25s` library**: it can return arbitrary metadata directly (`corpus=metas`), so a search result is already a source location.
- **One AST chunk per function or class**: it keeps identifiers, docstrings and code together, which is what a question about code is matched against.
- **No overlap between chunks**: simpler, and enough to pass the thresholds.
- **Separate commands linked by files**: the index is computed once; searching and answering do not need to rebuild it.
- **Thinking mode disabled in Qwen3**: answers are direct and about ten times faster.
- **Filename and class-name context**: the words of the file name, and the class name for a method pulled out of an oversized class, are prepended to the indexed text of a chunk. This gave the biggest recall gain of the project (docs 0.810 → 0.860, code 0.535 → 0.636), since many questions name a file or a class rather than quoting the code itself.

## Challenges faced

- **Oversized chunks**: 23% of Python units exceeded 2000 characters (one class was 150 000). Recursing into class methods, then raw slicing, fixed it, and no chunk exceeds the limit anymore.
- **Empty spans in Markdown**: a blank line right before a title produced zero-length chunks (76 of them). Empty ranges are now filtered out.
- **Code recall just under the threshold (0.485)**: the failing questions showed that the IoU of 0.05 penalizes a large chunk that contains a very small target (93 characters in a 2000-character chunk gives 0.046)

## Resources

- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), <https://arxiv.org/abs/2005.11401>
- `bm25s` library: <https://github.com/xhluca/bm25s>
- Python `ast` module: <https://docs.python.org/3/library/ast.html>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Hugging Face Transformers, Python Fire, pydantic and uv documentation

**Use of AI**: Claude was used throughout as a tutor/reviewer, not a code
generator — all code was written by hand. It was used to: explain concepts before writing code and point out concrete bugs and write this README
