"""여러 소스(it2_*.jsonl, it_sub_relabel_role.jsonl)에서 로드된 ITSubQueryRecord 병합 —
Architecture_Design.md 9.2절, P1_설계서_ITSubClassification.md 2.4절 참고.
등급 A(순수 로직) — 파일/포맷을 모르고 list[ITSubQueryRecord] 위에서만 동작한다.

라벨별/조합별 최소 건수 미달은 여기서 차단하지 않는다 — 요구사항정의서 4.5절 반복
보강 절차의 대상이며, 호출부(CLI)가 label_counts/combo_counts로 집계해 경고만 남긴다."""

from collections import Counter

from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.constants import SUB_LABELS
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


def combine(sources: list[list[ITSubQueryRecord]]) -> list[ITSubQueryRecord]:
    combined = [record for records in sources for record in records]

    seen: set[tuple[str, tuple[str, ...]]] = set()
    duplicates: set[tuple[str, tuple[str, ...]]] = set()
    for record in combined:
        key = (record.query, tuple(sorted(record.sub_categories)))
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    if duplicates:
        raise DataValidationError(f"중복 레코드 존재 (질의, 세부카테고리 조합): {sorted(duplicates)}")

    return combined


def label_counts(records: list[ITSubQueryRecord]) -> dict[str, int]:
    """라벨별 양성 건수 — 레코드가 여러 라벨에 속하면 각 라벨마다 1씩 카운트."""
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record.sub_categories)
    return {label: counts.get(label, 0) for label in SUB_LABELS}


def combo_counts(records: list[ITSubQueryRecord]) -> dict[str, int]:
    """라벨 2개 이상인 레코드만 대상으로, 정렬된 라벨 조합("DBA+MIDDLEWARE")별 건수."""
    counts: Counter[str] = Counter()
    for record in records:
        if len(record.sub_categories) >= 2:
            counts["+".join(sorted(record.sub_categories))] += 1
    return dict(counts)
