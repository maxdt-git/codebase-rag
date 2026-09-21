# VENV := /goinfre/$(USER)/rag/.venv
# UV_ENV := UV_PROJECT_ENVIRONMENT=$(VENV)

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

fclean: clean
	rm -rf .venv
	rm -rf data/processed data/output

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs \
	--strict


K = 5
MAX_CHUNK_SIZE = 2000
DATASET = dataset_docs_public.json
SEARCH_DIR = data/output/search_results/UnansweredQuestions
ANSWER_DIR = data/output/search_results_and_answer/UnansweredQuestions


index:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search "$(Q)" --k $(K)

search-dataset:
	uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/$(DATASET) \
	--k $(K) --save_directory $(SEARCH_DIR)

evaluate:
	uv run python -m src evaluate \
	--student_search_results_path $(SEARCH_DIR)/$(DATASET) \
	--dataset_path data/datasets/AnsweredQuestions/$(DATASET)

answer:
	uv run python -m src answer "$(Q)" --k $(K)

answer-dataset:
	uv run python -m src answer_dataset \
	--student_search_results_path $(SEARCH_DIR)/$(DATASET) \
	--save_directory $(ANSWER_DIR)


MOULINETTE = ./moulinette_pkg/moulinette-fedora
DOCS = dataset_docs_public.json
CODE = dataset_code_public.json


moulinette-docs:
	$(MAKE) search-dataset DATASET=$(DOCS) K=5
	$(MOULINETTE) evaluate_student_search_results \
	$(SEARCH_DIR)/$(DOCS) data/datasets/AnsweredQuestions/$(DOCS) \
	--k 5 --max_context_length 2000 --threshold 0.80

moulinette-code:
	$(MAKE) search-dataset DATASET=$(CODE) K=5
	$(MOULINETTE) evaluate_student_search_results \
	$(SEARCH_DIR)/$(CODE) data/datasets/AnsweredQuestions/$(CODE) \
	--k 5 --max_context_length 2000 --threshold 0.50

moulinette: moulinette-docs moulinette-code


.PHONY: install run debug clean fclean lint lint-strict
.PHONY: index search search-dataset evaluate answer answer-dataset
.PHONY: moulinette moulinette-docs moulinette-code
