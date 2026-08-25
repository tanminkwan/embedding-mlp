import json

import pytest

from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.data_generation.it_sub_jsonl_repository import (
    ITSubJsonlRepository,
)
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


class TestITSubJsonlRepositoryLoad:
    def test_loads_records_with_sub_categories(self, tmp_path):
        path = tmp_path / "it_sub_data.jsonl"
        path.write_text(
            json.dumps(
                {"질의": "q1", "응답": "r1", "카테고리": "IT", "세부카테고리": ["DBA", "MIDDLEWARE"]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        records = ITSubJsonlRepository().load(str(path))

        assert len(records) == 1
        assert records[0].query == "q1"
        assert records[0].sub_categories == ["DBA", "MIDDLEWARE"]

    def test_raises_on_missing_sub_category_key(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            json.dumps({"질의": "q1", "응답": "r1", "카테고리": "IT"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(DataValidationError):
            ITSubJsonlRepository().load(str(path))

    def test_raises_on_invalid_json_line(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text("not json\n", encoding="utf-8")

        with pytest.raises(DataValidationError):
            ITSubJsonlRepository().load(str(path))

    def test_raises_on_unknown_label(self, tmp_path):
        path = tmp_path / "broken.jsonl"
        path.write_text(
            json.dumps(
                {"질의": "q1", "응답": "r1", "카테고리": "IT", "세부카테고리": ["ETC"]},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        with pytest.raises(DataValidationError, match="필드 검증 실패"):
            ITSubJsonlRepository().load(str(path))

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "with_blank.jsonl"
        record_line = json.dumps(
            {"질의": "q1", "응답": "r1", "카테고리": "IT", "세부카테고리": ["DBA"]},
            ensure_ascii=False,
        )
        path.write_text(f"{record_line}\n\n{record_line}\n", encoding="utf-8")

        records = ITSubJsonlRepository().load(str(path))

        assert len(records) == 2


class TestITSubJsonlRepositorySave:
    def test_writes_category_it_fixed_and_reloads_identically(self, tmp_path):
        path = tmp_path / "out.jsonl"
        records = [
            ITSubQueryRecord(query="q1", response="r1", sub_categories=["OS", "NETWORK"]),
        ]

        ITSubJsonlRepository().save(records, str(path))

        raw = json.loads(path.read_text(encoding="utf-8").strip())
        assert raw["카테고리"] == "IT"
        assert raw["세부카테고리"] == ["OS", "NETWORK"]

        reloaded = ITSubJsonlRepository().load(str(path))
        assert reloaded == records

    def test_raises_when_output_already_exists(self, tmp_path):
        path = tmp_path / "out.jsonl"
        path.write_text("기존 내용\n", encoding="utf-8")

        with pytest.raises(DataValidationError, match="이미 존재"):
            ITSubJsonlRepository().save([], str(path))
