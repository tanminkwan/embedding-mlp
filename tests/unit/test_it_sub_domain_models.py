import pytest
from pydantic import ValidationError

from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


class TestITSubQueryRecord:
    def test_accepts_single_known_label(self):
        record = ITSubQueryRecord(query="q", response="r", sub_categories=["DBA"])
        assert record.sub_categories == ["DBA"]

    def test_accepts_multiple_known_labels(self):
        record = ITSubQueryRecord(
            query="q", response="r", sub_categories=["MIDDLEWARE", "DBA"]
        )
        assert record.sub_categories == ["MIDDLEWARE", "DBA"]

    def test_rejects_empty_sub_categories(self):
        with pytest.raises(ValidationError, match="at least 1 label"):
            ITSubQueryRecord(query="q", response="r", sub_categories=[])

    def test_rejects_unknown_label(self):
        with pytest.raises(ValidationError, match="unknown"):
            ITSubQueryRecord(query="q", response="r", sub_categories=["ETC"])

    def test_rejects_duplicate_labels(self):
        with pytest.raises(ValidationError, match="duplicate"):
            ITSubQueryRecord(query="q", response="r", sub_categories=["DBA", "DBA"])
