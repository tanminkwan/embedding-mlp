"""2차(IT 세부 분류) 전용 데이터 모델 — Architecture_Design.md 9.2절 참고.
등급 A(순수 로직) — 테스트 먼저.

1차 domain.models.QueryRecord는 category가 CLASS_LABELS 중 단일값 전제라
세부카테고리(멀티라벨 리스트)를 담을 수 없어 재사용 불가 — Scope_Definition.md
2.3절 정정, P1_설계서_ITSubClassification.md 2.1절 참고."""

from pydantic import BaseModel, field_validator

from embedding_lr.it_sub_classification.constants import SUB_LABELS


class ITSubQueryRecord(BaseModel):
    """it_sub_data.jsonl 레코드 1건에 대응. `category`("IT" 고정)는 필드로 두지 않는다 —
    JSONL 입출력 시에만 고정값으로 기록한다(ITSubJsonlRepository)."""

    query: str
    response: str
    sub_categories: list[str]

    @field_validator("sub_categories")
    @classmethod
    def _validate_sub_categories(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("sub_categories must have at least 1 label")
        if len(value) != len(set(value)):
            raise ValueError("sub_categories must not contain duplicate labels")
        unknown = sorted(set(value) - set(SUB_LABELS))
        if unknown:
            raise ValueError(f"sub_categories contains unknown label(s): {unknown}")
        return value
