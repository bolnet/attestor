"""Tests for MemoryAgentBench benchmark runner."""

import pytest

from attestor.mab import (
    normalize_answer,
    substring_exact_match,
    exact_match,
    token_f1,
    binary_recall,
    ruler_recall,
    max_over_ground_truths,
    chunk_text,
    chunk_text_overlap,
    score_question,
    _flatten_answers,
    _extract_answer,
    _is_exact_source,
    _get_budget_for_source,
    _extract_entities_from_context,
)


class TestNormalizeAnswer:
    def test_lowercase(self):
        assert normalize_answer("Hello World") == "hello world"

    def test_remove_articles(self):
        assert normalize_answer("the cat and a dog") == "cat and dog"

    def test_remove_punctuation(self):
        assert normalize_answer("hello, world!") == "hello world"

    def test_collapse_whitespace(self):
        assert normalize_answer("hello   world") == "hello world"

    def test_empty_string(self):
        assert normalize_answer("") == ""

    def test_only_articles(self):
        assert normalize_answer("a an the") == ""


class TestSubstringExactMatch:
    def test_exact(self):
        assert substring_exact_match("Paris", "Paris") is True

    def test_substring(self):
        assert substring_exact_match("The answer is Paris", "Paris") is True

    def test_no_match(self):
        assert substring_exact_match("London", "Paris") is False

    def test_case_insensitive(self):
        assert substring_exact_match("the answer is PARIS", "paris") is True

    def test_with_punctuation(self):
        assert substring_exact_match("It's Paris!", "Paris") is True


class TestExactMatch:
    def test_exact(self):
        assert exact_match("Paris", "paris") is True

    def test_not_exact(self):
        assert exact_match("The answer is Paris", "Paris") is False

    def test_with_articles(self):
        assert exact_match("the Paris", "Paris") is True

    def test_with_punctuation(self):
        assert exact_match("Paris.", "Paris") is True


class TestTokenF1:
    def test_perfect(self):
        f1, prec, rec = token_f1("hello world", "hello world")
        assert f1 == 1.0
        assert prec == 1.0
        assert rec == 1.0

    def test_partial_overlap(self):
        f1, prec, rec = token_f1("hello world foo", "hello world bar")
        assert 0 < f1 < 1
        assert prec > 0
        assert rec > 0

    def test_no_overlap(self):
        f1, prec, rec = token_f1("foo bar", "baz qux")
        assert f1 == 0.0

    def test_yes_no_mismatch(self):
        f1, prec, rec = token_f1("yes", "no")
        assert f1 == 0.0

    def test_yes_yes_match(self):
        f1, prec, rec = token_f1("yes", "yes")
        assert f1 == 1.0

    def test_empty_prediction(self):
        f1, prec, rec = token_f1("", "hello")
        assert f1 == 0.0


class TestBinaryRecall:
    def test_all_present(self):
        assert binary_recall("Paris is the capital of France", ["Paris", "France"]) == 1

    def test_partial(self):
        assert binary_recall("Paris is great", ["Paris", "France"]) == 0

    def test_case_insensitive(self):
        assert binary_recall("PARIS france", ["paris", "France"]) == 1

    def test_empty_elements(self):
        assert binary_recall("anything", []) == 1

    def test_single_element(self):
        assert binary_recall("The city is Paris", ["Paris"]) == 1


class TestRulerRecall:
    def test_all_present(self):
        assert ruler_recall("A B C", ["A", "B", "C"]) == 1.0

    def test_partial(self):
        assert ruler_recall("A B", ["A", "B", "C"]) == pytest.approx(2 / 3)

    def test_none_present(self):
        assert ruler_recall("X Y Z", ["P", "Q"]) == 0.0

    def test_empty_elements(self):
        assert ruler_recall("anything", []) == 0.0

    def test_case_insensitive(self):
        assert ruler_recall("abc DEF", ["ABC", "def"]) == 1.0


class TestMaxOverGroundTruths:
    def test_single_string(self):
        score = max_over_ground_truths(exact_match, "paris", "Paris")
        assert score == 1.0

    def test_list_of_strings(self):
        score = max_over_ground_truths(exact_match, "paris", ["Paris", "London"])
        assert score == 1.0

    def test_nested_list(self):
        score = max_over_ground_truths(exact_match, "paris", [["Paris"], ["London"]])
        assert score == 1.0

    def test_no_match(self):
        score = max_over_ground_truths(exact_match, "berlin", ["Paris", "London"])
        assert score == 0.0

    def test_empty_ground_truths(self):
        score = max_over_ground_truths(exact_match, "paris", [])
        assert score == 0.0

    def test_with_tuple_metric(self):
        score = max_over_ground_truths(
            lambda p, g: token_f1(p, g), "hello world", ["hello world"]
        )
        assert score == 1.0

    def test_with_bool_metric(self):
        score = max_over_ground_truths(
            substring_exact_match, "the answer is Paris", ["Paris"]
        )
        assert score == 1.0


class TestChunkText:
    def test_short_text(self):
        chunks = chunk_text("Hello world.", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_splits_on_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, chunk_size=5)  # Very small budget
        assert len(chunks) >= 2

    def test_preserves_content(self):
        text = "One. Two. Three. Four. Five."
        chunks = chunk_text(text, chunk_size=1000)
        recombined = " ".join(chunks)
        assert "One" in recombined
        assert "Five" in recombined

    def test_handles_no_sentence_boundaries(self):
        text = "word " * 500  # No sentence-ending punctuation
        chunks = chunk_text(text, chunk_size=50)
        assert len(chunks) >= 2

    def test_empty_text(self):
        chunks = chunk_text("")
        assert len(chunks) == 1
        assert chunks[0] == ""

    def test_newline_fallback(self):
        text = "Line one\nLine two\nLine three\nLine four"
        # Make it long enough to trigger newline fallback (>1000 chars)
        text = "\n".join([f"Line {i} " * 20 for i in range(20)])
        chunks = chunk_text(text, chunk_size=50)
        assert len(chunks) >= 2


class TestScoreQuestion:
    def test_ruler_qa(self):
        scores = score_question("The answer is Paris", ["Paris"], "ruler_qa")
        assert "substring_exact_match" in scores
        assert scores["substring_exact_match"] == 1.0

    def test_ruler_qa_miss(self):
        scores = score_question("London", ["Paris"], "ruler_qa")
        assert scores["substring_exact_match"] == 0.0

    def test_eventqa(self):
        scores = score_question("Paris France", ["Paris", "France"], "eventqa")
        assert "binary_recall" in scores
        assert scores["binary_recall"] == 1.0

    def test_eventqa_partial(self):
        scores = score_question("Paris only", ["Paris", "France"], "eventqa")
        assert scores["binary_recall"] == 0.0

    def test_factconsolidation(self):
        scores = score_question("Paris", ["Paris", "London"], "factconsolidation")
        assert "exact_match" in scores
        assert scores["exact_match"] == 1.0

    def test_icl(self):
        scores = score_question("cat", ["cat", "dog"], "icl_classify")
        assert scores["exact_match"] == 1.0

    def test_detective_qa(self):
        scores = score_question("butler", ["butler"], "detective_qa")
        assert scores["exact_match"] == 1.0

    def test_longmemeval_fallback(self):
        scores = score_question("Paris is great", ["Paris"], "longmemeval")
        assert "substring_exact_match" in scores
        assert scores["substring_exact_match"] == 1.0

    def test_infbench_fallback(self):
        scores = score_question("hello world", ["hello world"], "infbench")
        assert "token_f1" in scores
        assert scores["token_f1"] == 1.0

    def test_unknown_source_default(self):
        scores = score_question("Paris", ["Paris"], "unknown_task")
        assert "substring_exact_match" in scores


class TestFlattenAnswers:
    def test_flat(self):
        assert _flatten_answers(["a", "b"]) == ["a", "b"]

    def test_nested(self):
        assert _flatten_answers([["a", "b"], ["c"]]) == ["a", "b", "c"]

    def test_empty(self):
        assert _flatten_answers([]) == []


class TestExtractAnswer:
    def test_bold_answer(self):
        text = "Based on the passages, the answer is **Belgium**"
        assert _extract_answer(text) == "Belgium"

    def test_multiple_bold_takes_last(self):
        text = "**Charles Dickens** married **Catherine**. Country: **Belgium**"
        assert _extract_answer(text) == "Belgium"

    def test_answer_colon(self):
        text = "Answer: Italy"
        assert _extract_answer(text) == "Italy"

    def test_answer_colon_multiline(self):
        text = "Some reasoning here.\nAnswer: rugby"
        assert _extract_answer(text) == "rugby"

    def test_short_text_returned_as_is(self):
        assert _extract_answer("Belgium") == "Belgium"

    def test_short_phrase(self):
        assert _extract_answer("United Kingdom") == "United Kingdom"

    def test_last_line_fallback(self):
        text = "Based on the analysis of all the passages and context provided above, the relevant information indicates the following conclusion.\nBelgium"
        assert _extract_answer(text) == "Belgium"

    def test_strips_whitespace(self):
        assert _extract_answer("  Italy  ") == "Italy"

    def test_empty(self):
        assert _extract_answer("") == ""


class TestIsExactSource:
    def test_factconsolidation(self):
        assert _is_exact_source("factconsolidation_mh_6k") is True

    def test_memory_merging(self):
        assert _is_exact_source("memory_merging_200k") is True

    def test_icl(self):
        assert _is_exact_source("icl_classify") is True

    def test_detective(self):
        assert _is_exact_source("detective_qa") is True

    def test_ruler_qa(self):
        assert _is_exact_source("ruler_qa1_197K") is False

    def test_eventqa(self):
        assert _is_exact_source("eventqa_200k") is False


class TestChunkTextOverlap:
    def test_short_text_no_split(self):
        chunks = chunk_text_overlap("Hello world.", chunk_size=100)
        assert len(chunks) == 1
        assert chunks[0] == "Hello world."

    def test_splits_into_overlapping_chunks(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_text_overlap(text, chunk_size=5, overlap=2)
        assert len(chunks) >= 2
        # Overlapping chunks should share some content
        if len(chunks) >= 2:
            words_0 = set(chunks[0].split())
            words_1 = set(chunks[1].split())
            assert words_0 & words_1  # Some overlap

    def test_preserves_all_content(self):
        text = "One. Two. Three. Four. Five. Six. Seven. Eight."
        chunks = chunk_text_overlap(text, chunk_size=5, overlap=2)
        combined = " ".join(chunks)
        assert "One" in combined
        assert "Eight" in combined

    def test_empty_text(self):
        chunks = chunk_text_overlap("")
        assert chunks == [""]

    def test_word_level_fallback_with_stride(self):
        text = "word " * 500  # No sentence boundaries
        chunks = chunk_text_overlap(text, chunk_size=50, overlap=10)
        assert len(chunks) >= 2


class TestGetBudgetForSource:
    def test_multi_hop(self):
        assert _get_budget_for_source("factconsolidation_mh_6k", 4000) == 8000

    def test_memory_merging(self):
        assert _get_budget_for_source("memory_merging_200k", 4000) == 8000

    def test_eventqa(self):
        assert _get_budget_for_source("eventqa_full", 4000) == 6000

    def test_default(self):
        assert _get_budget_for_source("ruler_qa1_197K", 6000) == 6000

    def test_respects_higher_default(self):
        assert _get_budget_for_source("eventqa_full", 10000) == 10000


class TestExtractEntitiesFromContext:
    def test_extracts_proper_nouns(self):
        text = "Charles Dickens wrote Our Mutual Friend in London."
        entities = _extract_entities_from_context(text)
        assert "Charles Dickens" in entities or "Charles" in entities
        assert "London" in entities

    def test_skips_common_words(self):
        text = "The answer is based on However Therefore Moreover."
        entities = _extract_entities_from_context(text)
        assert "The" not in entities
        assert "However" not in entities
        assert "Therefore" not in entities

    def test_deduplicates(self):
        text = "Paris is great. Paris is lovely. Paris is wonderful."
        entities = _extract_entities_from_context(text)
        assert entities.count("Paris") == 1

    def test_limits_to_10(self):
        text = " ".join(f"Entity{chr(65+i)} is here." for i in range(20))
        entities = _extract_entities_from_context(text)
        assert len(entities) <= 10

    def test_empty_text(self):
        assert _extract_entities_from_context("") == []

    def test_no_proper_nouns(self):
        assert _extract_entities_from_context("all lowercase words here") == []
