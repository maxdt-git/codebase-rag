VENV := /goinfre/$(USER)/rag/.venv
UV_ENV := UV_PROJECT_ENVIRONMENT=$(VENV)

install:
	$(UV_ENV) uv sync

run:
	$(UV_ENV) uv run python -m src index

debug:
	$(UV_ENV) uv run python -m pdb -m src

clean:
	$(UV_ENV) find . -type d -name "__pycache__" -exec rm -rf {} +
	$(UV_ENV) find . -type d -name ".mypy_cache" -exec rm -rf {} +
	$(UV_ENV) find . -type d -name ".pytest_cache" -exec rm -rf {} +

fclean: clean
	cd /goinfre/$(USER)/ && rm -rf rag/
	$(UV_ENV) rm -rf data/processed data/output

lint:
	$(UV_ENV) uv run flake8 src
	$(UV_ENV) uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	$(UV_ENV) uv run flake8 src
	$(UV_ENV) uv run mypy src --warn-return-any --warn-unused-ignores \
	--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs \
	--strict


K = 5
MAX_CHUNK_SIZE = 2000
# SPLIT = private
DATASET = dataset_docs_$(SPLIT).json
SEARCH_DIR = data/output/search_results/UnansweredQuestions
ANSWER_DIR = data/output/search_results_and_answer/UnansweredQuestions
Q = What is vLLM


index:
	$(UV_ENV) uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	$(UV_ENV) uv run python -m src search "$(Q)" --k $(K)

search-dataset:
	$(UV_ENV) uv run python -m src search_dataset \
	--dataset_path data/datasets/UnansweredQuestions/$(DATASET) \
	--k $(K) --save_directory $(SEARCH_DIR)

evaluate:
	$(UV_ENV) uv run python -m src evaluate \
	--student_search_results_path $(SEARCH_DIR)/$(DATASET) \
	--dataset_path data/datasets/AnsweredQuestions/$(DATASET)

answer:
	$(UV_ENV) uv run python -m src answer "$(Q)" --k $(K)

answer-dataset:
	$(UV_ENV) uv run python -m src answer_dataset \
	--student_search_results_path $(SEARCH_DIR)/$(DATASET) \
	--save_directory $(ANSWER_DIR)


MOULINETTE = ./moulinette_pkg/moulinette-fedora
PUBPRI = public

moulinette-docs:
	$(MAKE) search-dataset SPLIT=$(PUBPRI) DATASET=dataset_docs_$(PUBPRI).json K=5
	$(MOULINETTE) evaluate_student_search_results \
	$(SEARCH_DIR)/dataset_docs_$(PUBPRI).json data/datasets/AnsweredQuestions/dataset_docs_$(PUBPRI).json \
	--k 5 --max_context_length 2000 --threshold 0.80

moulinette-code:
	$(MAKE) search-dataset SPLIT=$(PUBPRI) DATASET=dataset_code_$(PUBPRI).json K=5
	$(MOULINETTE) evaluate_student_search_results \
	$(SEARCH_DIR)/dataset_code_$(PUBPRI).json data/datasets/AnsweredQuestions/dataset_code_$(PUBPRI).json \
	--k 5 --max_context_length 2000 --threshold 0.50

moulinette: moulinette-docs moulinette-code

script:
	$(UV_ENV) ./exams/scripts/exam_retrieval.sh --student-path . --moulinette-path ./moulinette_pkg/moulinette-ubuntu


.PHONY: install run debug clean fclean lint lint-strict
.PHONY: index search search-dataset evaluate answer answer-dataset
.PHONY: moulinette moulinette-docs moulinette-code
.PHONY: script
