import pytest

from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.dataset.combine import combine, combo_counts, label_counts
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


def _record(query: str, labels: list[str], response: str = "r") -> ITSubQueryRecord:
    return ITSubQueryRecord(query=query, response=response, sub_categories=labels)


class TestCombine:
    def test_concatenates_all_sources(self):
        sources = [
            [_record("q1", ["DBA"])],
            [_record("q2", ["OS"]), _record("q3", ["NETWORK", "OS"])],
        ]

        combined = combine(sources)

        assert len(combined) == 3
        assert [r.query for r in combined] == ["q1", "q2", "q3"]

    def test_raises_when_same_query_and_label_set_duplicated(self):
        sources = [
            [_record("dup", ["DBA", "MIDDLEWARE"])],
            [_record("dup", ["MIDDLEWARE", "DBA"])],  # 순서만 다름 — 같은 조합으로 취급
        ]

        with pytest.raises(DataValidationError, match="중복"):
            combine(sources)

    def test_allows_same_query_with_different_label_set(self):
        sources = [
            [_record("same query", ["DBA"])],
            [_record("same query", ["OS"])],
        ]

        combined = combine(sources)

        assert len(combined) == 2


class TestLabelCounts:
    def test_counts_each_label_occurrence_across_records(self):
        records = [
            _record("q1", ["DBA"]),
            _record("q2", ["DBA", "MIDDLEWARE"]),
            _record("q3", ["OS"]),
        ]

        counts = label_counts(records)

        assert counts["DBA"] == 2
        assert counts["MIDDLEWARE"] == 1
        assert counts["OS"] == 1
        assert counts["NETWORK"] == 0
        assert counts["DEVOPS"] == 0


class TestComboCounts:
    def test_counts_multi_label_combinations_only(self):
        records = [
            _record("q1", ["DBA"]),
            _record("q2", ["DBA", "MIDDLEWARE"]),
            _record("q3", ["MIDDLEWARE", "DBA"]),  # 순서만 다름 — 같은 조합
            _record("q4", ["OS"]),
        ]

        counts = combo_counts(records)

        assert counts == {"DBA+MIDDLEWARE": 2}
