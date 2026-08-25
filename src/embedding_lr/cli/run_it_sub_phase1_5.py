"""Phase 1.5(2차) CLI — it2_*.jsonl + it_sub_relabel_*.jsonl 병합 + 반복 계층화 분할.
Architecture_Design.md 9.2절, P1_설계서_ITSubClassification.md 4절 참고.

Trigger: python -m embedding_lr.cli.run_it_sub_phase1_5 --input-dir <path> --output-dir <path>
Input:   --input-dir(it2_*.jsonl 여러 개 + it_sub_relabel_*.jsonl이 있는 디렉터리)
Output:  --output-dir(it_sub_data.jsonl, it_sub_{train,test,val}.jsonl)
         — 각 파일이 이미 존재하면 실패(입출력 보존 원칙)
"""

import argparse
from pathlib import Path

from embedding_lr.config import Settings
from embedding_lr.it_sub_classification.constants import (
    MIN_COMBO_POSITIVE_COUNT,
    MIN_LABEL_POSITIVE_COUNT,
    SUB_LABELS,
)
from embedding_lr.it_sub_classification.data_generation.it_sub_jsonl_repository import (
    ITSubJsonlRepository,
)
from embedding_lr.it_sub_classification.dataset.combine import combine, combo_counts, label_counts
from embedding_lr.it_sub_classification.dataset.split import iterative_stratified_split
from embedding_lr.logging_config import setup_logging
from embedding_lr.workflow.run_context import run_context

_SPLIT_STEMS = {"train": "it_sub_train", "test": "it_sub_test", "validation": "it_sub_val"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1.5(2차): 병합 + 반복 계층화 분할")
    parser.add_argument(
        "--input-dir", required=True, help="it2_*.jsonl + it_sub_relabel_*.jsonl이 있는 디렉터리"
    )
    parser.add_argument(
        "--output-dir", required=True, help="it_sub_data/train/test/val.jsonl 출력 디렉터리"
    )
    args = parser.parse_args()

    settings = Settings()
    setup_logging(settings)

    with run_context("it_sub_phase1_5", settings) as (run_id, logger):
        repo = ITSubJsonlRepository()
        input_dir = Path(args.input_dir)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        source_paths = sorted(input_dir.glob("it2_*.jsonl")) + sorted(
            input_dir.glob("it_sub_relabel_*.jsonl")
        )
        logger.info("소스 파일 로드 시작", extra={"extra": {"count": len(source_paths)}})
        sources = [repo.load(str(p)) for p in source_paths]

        combined = combine(sources)
        data_path = output_dir / "it_sub_data.jsonl"
        repo.save(combined, str(data_path))
        logger.info(
            "병합 완료", extra={"extra": {"count": len(combined), "output": str(data_path)}}
        )

        for label, count in label_counts(combined).items():
            if count < MIN_LABEL_POSITIVE_COUNT:
                logger.warning(
                    "라벨별 최소 건수 미달", extra={"extra": {"label": label, "count": count}}
                )
        for combo, count in combo_counts(combined).items():
            if count < MIN_COMBO_POSITIVE_COUNT:
                logger.warning(
                    "라벨 조합별 최소 건수 미달", extra={"extra": {"combo": combo, "count": count}}
                )

        train, test, val = iterative_stratified_split(combined)
        named_splits = (("train", train), ("test", test), ("validation", val))

        for name, records in named_splits:
            missing = [label for label in SUB_LABELS if label_counts(records)[label] == 0]
            if missing:
                logger.warning(
                    "split에 라벨 커버리지 누락",
                    extra={"extra": {"split": name, "missing": missing}},
                )

        for name, records in named_splits:
            split_path = output_dir / f"{_SPLIT_STEMS[name]}.jsonl"
            repo.save(records, str(split_path))
            logger.info(
                "분할 저장 완료",
                extra={
                    "extra": {"split": name, "count": len(records), "output": str(split_path)}
                },
            )


if __name__ == "__main__":
    main()
