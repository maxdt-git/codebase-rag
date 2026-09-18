from src.models import MinimalSource, MinimalAnswer
from src.models import StudentSearchResults, StudentSearchResultsAndAnswer
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from tqdm import tqdm


def get_source_text(source: MinimalSource) -> str:
    """
    Read the chunk text pointed to by a source location
    """
    content = Path(source.file_path).read_text(encoding="utf-8")
    return content[source.first_character_index:source.last_character_index]


def build_prompt(
        question: str,
        sources: list[MinimalSource]
) -> str:
    """
    Build the prompt with questionn and retrieve context
    """
    context = "\n\n".join(get_source_text(source) for source in sources)
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer the question using the context above."
        f"If the context doesn't contain the answer, say so."
    )


MODEL_NAME = "Qwen/Qwen3-0.6B"


def load_model() -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """
    Load the qwen tokenizer and model
    """
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def generate_answer(
        question: str,
        sources: list[MinimalSource],
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel
) -> str:
    """
    Generate an answer grounded in the retrieved sources
    """
    prompt = build_prompt(question, sources)
    messages = [{"role": "user", "content": prompt}]
    # enable_thinking=False: skip Qwen3's reasoning block, answer directly
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer(text, return_tensors="pt")
    # do_sample=False - same prompt always gives same answer
    output = model.generate(  # type: ignore[operator]
        **inputs, max_new_tokens=300, do_sample=False
    )
    # keep only the newly generated tokens, drop the prompt itself
    generated = output[0][inputs["input_ids"].shape[1]:]
    return str(tokenizer.decode(generated, skip_special_tokens=True))


def answer_dataset(
        student_results: StudentSearchResults,
        tokenizer: PreTrainedTokenizerBase,
        model: PreTrainedModel
) -> StudentSearchResultsAndAnswer:
    """
    Generate an answer for every question of a StudentSearchResults
    """
    minimal_answers: list[MinimalAnswer] = []
    # reuse the sources search_dataset already found, no re-search here
    for result in tqdm(student_results.search_results, desc="Answering"):
        answer = generate_answer(
            result.question, result.retrieved_sources, tokenizer, model
        )
        minimal_answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=answer
            )
        )
    return StudentSearchResultsAndAnswer(
        search_results=minimal_answers, k=student_results.k
    )


# if __name__ == "__main__":
#     import json

#     with open(
#         "data/output/search_results/UnansweredQuestions/"
#         "dataset_docs_public.json"
#     ) as f:
#         data = json.load(f)

#     data["search_results"] = data["search_results"][:5]
#     student_results = StudentSearchResults.model_validate(data)

#     tokenizer, model = load_model()
#     result = answer_dataset(student_results, tokenizer, model)

#     for r in result.search_results:
#         print("QUESTION:", r.question)
#         print("REPONSE:", r.answer)
#         print()
