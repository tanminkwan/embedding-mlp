import json

import pytest

from embedding_lr.cli.run_it_sub_phase1_5 import main
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


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for query, labels in records:
            f.write(
                json.dumps(
                    {"질의": query, "응답": "r", "카테고리": "IT", "세부카테고리": labels},
                    ensure_ascii=False,
                )
                + "\n"
            )


def _seed_input_dir(input_dir):
    input_dir.mkdir(parents=True, exist_ok=True)
    labels = ["DBA", "DEVOPS", "MIDDLEWARE", "NETWORK", "OS"]
    for label in labels:
        _write_jsonl(
            input_dir / f"it2_{label.lower()}.jsonl",
            [(f"{label}-single-{i}", [label]) for i in range(15)],
        )
    _write_jsonl(
        input_dir / "it2_pair_middleware_dba.jsonl",
        [(f"combo-mw-dba-{i}", ["MIDDLEWARE", "DBA"]) for i in range(10)],
    )
    _write_jsonl(
        input_dir / "it_sub_relabel_role.jsonl",
        [(f"relabel-{i}", ["OS", "NETWORK"]) for i in range(10)],
    )


class TestRunItSubPhase1_5:
    def test_combines_and_splits_all_sources(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _seed_input_dir(input_dir)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_it_sub_phase1_5",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        main()

        repo = ITSubJsonlRepository()
        combined = repo.load(str(output_dir / "it_sub_data.jsonl"))
        assert len(combined) == 15 * 5 + 10 + 10

        train = repo.load(str(output_dir / "it_sub_train.jsonl"))
        test = repo.load(str(output_dir / "it_sub_test.jsonl"))
        val = repo.load(str(output_dir / "it_sub_val.jsonl"))
        assert len(train) + len(test) + len(val) == len(combined)

        all_queries = sorted(r.query for r in train + test + val)
        assert all_queries == sorted(r.query for r in combined)

    def test_records_succeeded_status(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _seed_input_dir(input_dir)

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_it_sub_phase1_5",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        main()

        status_files = list((tmp_path / "status").glob("it_sub_phase1_5_*.json"))
        assert len(status_files) == 1
        status = json.loads(status_files[0].read_text(encoding="utf-8"))
        assert status["status"] == "succeeded"

    def test_succeeds_even_when_a_label_cannot_cover_every_split(self, monkeypatch, tmp_path):
        # DEVOPS가 1건뿐이라 3개 split 중 최소 2곳에는 커버리지가 없을 수밖에 없다 —
        # 이 상황에서도 CLI는 실패하지 않고 경고만 남기고 정상 종료해야 한다.
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir(parents=True)
        _write_jsonl(
            input_dir / "it2_dba.jsonl", [(f"dba-{i}", ["DBA"]) for i in range(15)]
        )
        _write_jsonl(input_dir / "it2_sparse.jsonl", [("devops-only", ["DEVOPS"])])

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_it_sub_phase1_5",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        main()  # 예외 없이 종료되어야 함

        repo = ITSubJsonlRepository()
        combined = repo.load(str(output_dir / "it_sub_data.jsonl"))
        assert any(r.query == "devops-only" for r in combined)

    def test_raises_when_output_already_exists(self, monkeypatch, tmp_path):
        _set_env(monkeypatch, tmp_path)
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        _seed_input_dir(input_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "it_sub_data.jsonl").write_text("기존 내용\n", encoding="utf-8")

        monkeypatch.setattr(
            "sys.argv",
            [
                "run_it_sub_phase1_5",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
            ],
        )

        with pytest.raises(Exception, match="이미 존재"):
            main()
