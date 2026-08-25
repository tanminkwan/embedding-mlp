import json

import pytest

from embedding_lr.cli.run_it_sub_phase1 import main
from embedding_lr.it_sub_classification.data_generation.it_sub_jsonl_repository import (
    ITSubJsonlRepository,
)


REQUIRED_ENV = {
    "AIPRO_BASE_URL": "http://localhost:28000",
    "AIPRO_API_TOKEN": "test-token",
    "EMBEDDING_SERVER_BASE_URL": "http://localhost:8000",
    "MODEL_DIR": "models",
    "MODEL_PATH": "models/model.pkl",
}


def _set_env(monkeypatch, tmp_path):
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("STATUS_DIR", str(tmp_path / "status"))


class TestRunItSubPhase1:
    def test_converts_it2_csv_to_jsonl(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "it2_pair_middleware_dba.csv"
        output_path = tmp_path / "it2_pair_middleware_dba.jsonl"
        input_path.write_text(
            '"질의","응답","카테고리"\n"q1","r1","MIDDLEWARE+DBA"\n', encoding="utf-8"
        )

        monkeypatch.setattr(
            "sys.argv",
            ["run_it_sub_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        records = ITSubJsonlRepository().load(str(output_path))
        assert len(records) == 1
        assert records[0].query == "q1"
        assert records[0].sub_categories == ["MIDDLEWARE", "DBA"]

    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "it2_dba.csv"
        output_path = tmp_path / "it2_dba.jsonl"
        input_path.write_text('"질의","응답","카테고리"\n"q1","r1","DBA"\n', encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            ["run_it_sub_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        main()

        status_files = list((tmp_path / "status").glob("it_sub_phase1_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"

    def test_raises_when_output_already_exists(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_path = tmp_path / "it2_dba.csv"
        output_path = tmp_path / "it2_dba.jsonl"
        input_path.write_text('"질의","응답","카테고리"\n"q1","r1","DBA"\n', encoding="utf-8")
        output_path.write_text("기존 내용\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            ["run_it_sub_phase1", "--input", str(input_path), "--output", str(output_path)],
        )

        with pytest.raises(Exception, match="이미 존재"):
            main()
