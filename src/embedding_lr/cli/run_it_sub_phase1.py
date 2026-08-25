"""Phase 1(2차) CLI — it2_*.csv를 정규화 JSONL로 변환. Architecture_Design.md 9.2절,
P1_설계서_ITSubClassification.md 4절 참고.

Trigger: python -m embedding_lr.cli.run_it_sub_phase1 --input <it2 csv path> --output <jsonl path>
Input:   --input(it2_*.csv, 카테고리="LABEL" 또는 "LABEL1+LABEL2" 형식)
Output:  --output(변환된 JSONL 파일) — 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse

from embedding_lr.config import Settings
from embedding_lr.it_sub_classification.data_generation.it2_csv_repository import It2CsvRepository
from embedding_lr.it_sub_classification.data_generation.it_sub_jsonl_repository import (
    ITSubJsonlRepository,
)
from embedding_lr.logging_config import setup_logging
from embedding_lr.workflow.run_context import run_context


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1(2차): it2_*.csv → JSONL 변환")
    parser.add_argument("--input", required=True, help="변환할 it2_*.csv 파일 경로")
    parser.add_argument("--output", required=True, help="출력 JSONL 파일 경로")
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("it_sub_phase1", settings) as (run_id, logger):
        logger.info("it2 CSV → JSONL 변환 시작", extra={"extra": {"input": args.input}})
        records = It2CsvRepository().load(args.input)
        ITSubJsonlRepository().save(records, args.output)
        logger.info(
            "it2 CSV → JSONL 변환 완료",
            extra={"extra": {"count": len(records), "output": args.output}},
        )


if __name__ == "__main__":
    main()
