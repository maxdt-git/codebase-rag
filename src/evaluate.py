from src.models import RagDataset, AnsweredQuestion
from src.models import MinimalSource, StudentSearchResults


def iou(
        a1: int,
        b1: int,
        a2: int,
        b2: int
) -> float:
    """
    Compute the intersection over union of two character range
    """
    # overlap between the two ranges, 0 if they don't touch at all
    intersection = max(0, min(b1, b2) - max(a1, a2))
    # total length covered by both ranges combined
    union = (b1 - a1) + (b2 - a2) - intersection

    if union > 0:
        return intersection / union
    else:
        return 0.0


def build_ground_truth(
        dataset: RagDataset
) -> dict[str, list[MinimalSource]]:
    """
    Map each answered question's id to its ground-truth sources
    """
    results: dict[str, list[MinimalSource]] = {}
    # only AnsweredQuestion has sources, UnansweredQuestion is skipped
    for question in dataset.rag_questions:
        if isinstance(question, AnsweredQuestion):
            results[question.question_id] = question.sources

    return results


def compute_recall(
        student_results: StudentSearchResults,
        ground_truth: dict[str, list[MinimalSource]]
) -> float:
    """
    Comnpute the average recall over all evaluated questions
    """
    total_recall = 0.0
    count = 0

    for result in student_results.search_results:
        # skip a question we have no ground truth to compare against
        true_sources = ground_truth.get(result.question_id)
        if not true_sources:
            continue

        found = 0
        # a true source counts as found if any retrieved source overlaps it
        for true_source in true_sources:
            hit = False
            for r in result.retrieved_sources:
                if r.file_path == true_source.file_path:
                    overlap = iou(
                        r.first_character_index,
                        r.last_character_index,
                        true_source.first_character_index,
                        true_source.last_character_index
                    )
                    if overlap > 0.05:
                        hit = True
                        break

            if hit:
                found += 1

        # recall for this question: fraction of its sources we found
        total_recall += found / len(true_sources)
        count += 1

    return total_recall / count if count > 0 else 0.0


# if __name__ == "__main__":
#     with open(
#         "data/datasets/AnsweredQuestions/dataset_docs_public.json",
#         encoding="utf-8") as f:
#         dataset = RagDataset.model_validate(json.load(f))
#     ground_truth = build_ground_truth(dataset)

#     with open(
#         "data/output/search_results/UnansweredQuestions/" \
#         "dataset_docs_public.json", encoding="utf-8") as f:
#         student_results = StudentSearchResults.model_validate(json.load(f))

#     recall = compute_recall(student_results, ground_truth)
#     print(f"recall@k = {recall:.3f}")
