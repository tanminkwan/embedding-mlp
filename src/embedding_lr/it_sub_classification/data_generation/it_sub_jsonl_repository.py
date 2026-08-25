"""ITSubDataRepository 구현체(JSONL) — Architecture_Design.md 9.2절 참고.
등급 B(오케스트레이션, 파일 I/O) — 구현 후 통합 테스트."""

import json
import os

from pydantic import ValidationError as PydanticValidationError

from embedding_lr.constants import FIELD_CATEGORY, FIELD_QUERY, FIELD_RESPONSE
from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.constants import FIELD_SUB_CATEGORY
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


class ITSubJsonlRepository:
    """it_sub 스키마(질의/응답/카테고리="IT" 고정/세부카테고리) JSONL 읽기/쓰기.
    `category`는 도메인 모델(ITSubQueryRecord)에 필드로 두지 않으므로, 여기서만
    "IT" 고정값으로 기록하고 로드 시에는 읽지 않는다(2차 대상은 항상 IT)."""

    def load(self, path: str) -> list[ITSubQueryRecord]:
        records: list[ITSubQueryRecord] = []
        with open(path, encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataValidationError(f"{path}:{line_number} JSON 파싱 실패: {exc}") from exc
                try:
                    records.append(
                        ITSubQueryRecord(
                            query=raw[FIELD_QUERY],
                            response=raw[FIELD_RESPONSE],
                            sub_categories=raw[FIELD_SUB_CATEGORY],
                        )
                    )
                except KeyError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필수 키 누락: {exc}") from exc
                except PydanticValidationError as exc:
                    raise DataValidationError(f"{path}:{line_number} 필드 검증 실패: {exc}") from exc
        return records

    def save(self, records: list[ITSubQueryRecord], path: str) -> None:
        if os.path.exists(path):
            raise DataValidationError(f"{path} 이미 존재 — 덮어쓰기 금지(입출력 보존 원칙)")
        with open(path, "w", encoding="utf-8") as f:
            for record in records:
                line = json.dumps(
                    {
                        FIELD_QUERY: record.query,
                        FIELD_RESPONSE: record.response,
                        FIELD_CATEGORY: "IT",
                        FIELD_SUB_CATEGORY: record.sub_categories,
                    },
                    ensure_ascii=False,
                )
                f.write(line + "\n")
