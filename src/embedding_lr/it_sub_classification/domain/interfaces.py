"""2차(IT 세부 분류) 전용 Protocol 정의 — Architecture_Design.md 9.2절 참고.

1차 domain.interfaces.DataRepository는 반환 타입이 list[QueryRecord]로 고정돼 있어
세부카테고리(멀티라벨)를 담는 레코드를 반환할 수 없어 재사용 불가 — Scope_Definition.md
2.3절 정정, P1_설계서_ITSubClassification.md 2.2절 참고."""

from typing import Protocol

from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


class ITSubDataRepository(Protocol):
    def load(self, path: str) -> list[ITSubQueryRecord]: ...

    def save(self, records: list[ITSubQueryRecord], path: str) -> None: ...
