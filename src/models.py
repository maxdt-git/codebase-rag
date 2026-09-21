import uuid
from pydantic import BaseModel, Field
from typing import List


class MinimalSource(BaseModel):
    """
    A source location: a file path and a character range
    """
    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """
    A question without its answer or sources
    """
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """
    A question with its ground-truth sources and answer
    """
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """
    A dataset of answered or unanswered questions
    """
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """
    The sources retrieved for one question
    """
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """
    The retrieved sources and the generated answer for one question
    """
    answer: str


class StudentSearchResults(BaseModel):
    """
    The output of search_dataset: search results for a whole dataset
    """
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """
    The output of answer_dataset: search results with their answers
    """
    search_results: List[MinimalAnswer]
    k: int
