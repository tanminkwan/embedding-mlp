"""ITSubDataRepository 구현체 — it2_*.csv(카테고리="LABEL1+LABEL2" 형식) 읽기 전용.
Architecture_Design.md 9.2절, P1_설계서_ITSubClassification.md 2.3절 참고.
등급 B(오케스트레이션, 파일 I/O) — 구현 후 통합 테스트."""

import csv

from pydantic import ValidationError as PydanticValidationError

from embedding_lr.constants import FIELD_CATEGORY, FIELD_QUERY, FIELD_RESPONSE
from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


class It2CsvRepository:
    """it2_*.csv 원본 읽기 전용 — 카테고리 필드를 "+"로 분해해 세부카테고리 리스트로 변환.
    파일명(it2_pair_middleware_dba.csv 등)은 참고용일 뿐 라벨 판단에 쓰지 않는다 —
    카테고리 필드값만을 유일한 정답 소스로 취급한다."""

    def load(self, path: str) -> list[ITSubQueryRecord]:
        records: list[ITSubQueryRecord] = []
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for line_number, row in enumerate(reader, start=2):  # 1행 = 헤더
                try:
                    labels = row[FIELD_CATEGORY].split("+")
                    records.append(
                        ITSubQueryRecord(
                            query=row[FIELD_QUERY],
                            response=row[FIELD_RESPONSE],
                            sub_categories=labels,
                        )
                    )
                except KeyError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필수 컬럼 누락: {exc}") from exc
                except PydanticValidationError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필드 검증 실패: {exc}") from exc
        return records

    def save(self, records: list[ITSubQueryRecord], path: str) -> None:
        raise NotImplementedError(
            "it2 CSV 저장은 지원하지 않습니다 — 이 프로젝트는 JSONL만 출력합니다"
            "(ITSubJsonlRepository 사용)."
        )
