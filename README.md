*This activity has been created as part of the 42 curriculum by mde-nas.*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) system that answers questions about a codebase, here the [vLLM](https://github.com/vllm-project/vllm) 0.10.1 repository.

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

`make run` and `make debug` start the CLI normally or under `pdb`. Shortcuts exist for the whole pipeline (`search`, `search-dataset`, `evaluate`, `answer`, `answer-dataset`); variables `K`, `MAX_CHUNK_SIZE`, `DATASET` and `Q` can be overridden, e.g. `make evaluate K=5 DATASET=dataset_code_public.json`.

**CLI** (built with Python Fire, `uv run python -m src <command>`):

| Command | Options | Effect |
|---|---|---|
| `index` | `--max_chunk_size` (1 to 2000, default 2000) | Index `data/raw/` into `data/processed/` |
| `search <query>` | `--k` | Print the top-k sources of one query |
| `search_dataset` | `--dataset_path --k --save_directory` | Search a whole dataset, write a `StudentSearchResults` JSON |
| `answer <query>` | `--k` | Search, then answer one query |
| `answer_dataset` | `--student_search_results_path --save_directory` | Answer every question of a search-results file, write a `StudentSearchResultsAndAnswer` JSON |
| `evaluate` | `--student_search_results_path --dataset_path` | Print recall@k against a ground-truth dataset |

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
# recall@k = 0.810

uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

The official moulinette provided with the subject (renamed `moulinette`) gives the same figures:

```bash
./moulinette evaluate_student_search_results \
  data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  data/datasets/AnsweredQuestions/dataset_docs_public.json \
  --k 5 --max_context_length 2000 --threshold 0.80
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

Every chunk is a character range `[start, end)` of the original file, never longer than `--max_chunk_size`. Two strategies are used, chosen by file extension.

**Python (`.py`)**: the file is parsed with `ast`, and each top-level function or class is one chunk, so a signature stays with its docstring and body. Line/column positions from the AST are converted to character offsets. If a unit is too big:
- a class is split into its methods (recursively);
- a function or method that is still too big is cut into consecutive slices of `max_chunk_size` characters (no overlap).

**Markdown and text (`.md`, `.txt`)**: the file is split at title lines (`#` to `######`). A section that is too big is split into paragraphs (blank-line separated), and a paragraph that is still too big is cut into raw slices. A file without titles, such as `CMakeLists.txt`, is one section.

Result: 1965 files give 21 973 chunks, none longer than 2000 characters.

## Retrieval method

- **Algorithm**: BM25 (`bm25s` library), one "document" per chunk. Chunk and query texts are lowercased, split into words and stripped of English stop words.
- **Ranking**: `retrieve` scores every chunk and returns the top-k. Chunks with a score of 0 (no word in common with the question) are dropped, so an empty or nonsensical query returns nothing instead of random chunks. `k` is capped to the number of chunks.
- **Parameters**: `k1=1.2` and `b=0.5` (library defaults are 1.5 and 0.75), tuned on the public datasets (see below).
- **Persistence**: the index and one metadata line per chunk (`file_path` and character range) are saved in `data/processed/`. Chunk texts are not stored; for generation, the text is re-read from the corpus with the character range.
- **Generation**: the retrieved chunks are concatenated as context, followed by the question and an instruction to answer only from the context. `Qwen/Qwen3-0.6B` runs with its thinking mode disabled and greedy decoding (deterministic), 300 new tokens at most.

## Performance analysis

Measured on the public datasets (100 docs questions, 99 code questions), and confirmed by the official moulinette.

| Recall | @1 | @3 | @5 | @10 | Required @5 |
|---|---|---|---|---|---|
| docs | 0.520 | 0.740 | **0.810** | 0.860 | 0.80 |
| code | 0.313 | 0.465 | **0.535** | 0.576 | 0.50 |

**Effect of `max_chunk_size`** (recall@5, `k1=1.2`, `b=0.5`):

| max_chunk_size | chunks | docs | code |
|---|---|---|---|
| **2000** | 21 973 | **0.810** | **0.535** |
| 1500 | 24 974 | 0.770 | 0.505 |
| 1000 | 30 596 | 0.770 | 0.455 |
| 500 | 46 874 | 0.710 | 0.424 |

Smaller chunks lower the recall in every case: the chunks split from a large function or section lose the context that made it match the question.

**Effect of BM25 parameters** (recall@5, chunk size 2000):

| k1 / b | docs | code |
|---|---|---|
| 1.5 / 0.75 (defaults) | 0.810 | 0.485 |
| **1.2 / 0.5** | **0.810** | **0.535** |

**System performance**:

| | Measured | Limit |
|---|---|---|
| Indexing of the whole corpus | about 5 s | 5 min |
| Retrieval, 199 questions | under 1 s | 90 s |
| Answer generation (CPU) | about 22 s per question | none in the subject |

## Design decisions

- **BM25 instead of TF-IDF**: BM25 saturates the effect of repeated words and normalizes by chunk length, which suits chunks of very different sizes.
- **`bm25s` library**: it can return arbitrary metadata directly (`corpus=metas`), so a search result is already a source location.
- **One AST chunk per function or class**: it keeps identifiers, docstrings and code together, which is what a question about code is matched against.
- **No overlap between chunks**: simpler, and enough to pass the thresholds.
- **Separate commands linked by files**: the index is computed once; searching and answering do not need to rebuild it.
- **Thinking mode disabled in Qwen3**: answers are direct and about ten times faster.
- **Known limitation**: top-level constants, imports and class docstrings are not indexed. Some code questions target them and are missed.

## Challenges faced

- **Character offsets**: the AST gives line and column numbers, but sources need character positions. A `line_offsets` helper converts them, and offsets are carried through every recursive split.
- **Oversized chunks**: 23% of Python units exceeded 2000 characters (one class was 150 000). Recursing into class methods, then raw slicing, fixed it, and no chunk exceeds the limit anymore.
- **Empty spans in Markdown**: a blank line right before a title produced zero-length chunks (76 of them). Empty ranges are now filtered out.
- **Code recall just under the threshold (0.485)**: the failing questions showed that the IoU of 0.05 penalizes a large chunk that contains a very small target (93 characters in a 2000-character chunk gives 0.046). Indexing every top-level statement as its own chunk (44 990 chunks) made things worse (docs 0.750, code 0.434), because tiny chunks are favored by BM25's length normalization. Tuning `k1` and `b` fixed it without changing the chunking.
- **Qwen3 thinking mode**: the model writes a reasoning block before the answer, slow and noisy. It is disabled with `enable_thinking=False`.
- **Degenerate inputs**: Fire converts arguments by itself (`search 123` gives an integer), `--max_chunk_size 0` made the splitter loop forever, and an empty query returned random chunks. Arguments are now validated, and score-0 results are dropped.

## Resources

- Robertson and Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009)
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (2020), <https://arxiv.org/abs/2005.11401>
- `bm25s` library: <https://github.com/xhluca/bm25s>
- Python `ast` module: <https://docs.python.org/3/library/ast.html>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- Hugging Face Transformers, Python Fire, pydantic and uv documentation

**Use of AI**: Claude Code (Anthropic) was used throughout as a tutor and pair-programmer. It explained the concepts (AST, regular expressions, BM25, pydantic, Fire, chat templates), guided the implementation step by step with code snippets that were typed, run and adapted, helped find bugs by running the code on the real corpus, wrote the measurement scripts used in the performance analysis, and wrote the docstrings, inline comments, Makefile shortcuts and this README. Every result reported here was measured by running the code, and the recall figures were checked with the official moulinette.
