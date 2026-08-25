"""ITSubQueryRecord 목록을 반복 계층화(iterative stratification)로 3:1:1 분할 —
Architecture_Design.md 9.2절, P1_설계서_ITSubClassification.md 3절 참고.
등급 A(순수 로직) — 동일 입력·시드 → 동일 출력.

1차 dataset/split.py(클래스당 단순 stratified 3:1:1)는 레코드당 라벨 1개 전제라
멀티라벨에 그대로 쓸 수 없어 별도 구현. 외부 라이브러리(scikit-multilearn 등) 없이
직접 구현한다 — 근거는 설계서 3.1절 참고. Sechidis et al.(2011) 반복 계층화의
단순화 버전: 매 단계 "현재 남은 후보 중 가장 희소한 라벨"을 우선 배정한다."""

import random
from collections import Counter

from embedding_lr.constants import RANDOM_SEED
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord

SplitGroups = tuple[list[ITSubQueryRecord], list[ITSubQueryRecord], list[ITSubQueryRecord]]


def iterative_stratified_split(
    records: list[ITSubQueryRecord],
    ratios: tuple[int, int, int] = (3, 1, 1),
    seed: int = RANDOM_SEED,
) -> SplitGroups:
    """반환값 순서: (train, test, validation)."""
    ratio_total = sum(ratios)

    rng = random.Random(seed)
    pool = list(records)
    rng.shuffle(pool)

    label_totals: Counter[str] = Counter()
    for record in pool:
        label_totals.update(record.sub_categories)

    # remaining_desired[label][split_index] — 라벨별 각 split에 아직 배정 가능한 여유(연속값)
    remaining_desired: dict[str, list[float]] = {
        label: [total * ratio / ratio_total for ratio in ratios]
        for label, total in label_totals.items()
    }

    assigned: list[list[ITSubQueryRecord]] = [[], [], []]
    assigned_count = [0, 0, 0]

    while pool:
        pool_label_counts: Counter[str] = Counter()
        for record in pool:
            pool_label_counts.update(record.sub_categories)
        rarest_label = min(pool_label_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        candidates = [r for r in pool if rarest_label in r.sub_categories]
        # 셔플된 pool 순서 그대로 첫 후보를 선택한다 — 라벨 수(단일/복합) 기준으로
        # 우선순위를 매기면 특정 라벨의 단일 라벨 레코드가 그 라벨의 목표 예산을
        # 먼저 다 써버려, 이후 처리되는 복합 라벨 레코드가 예산이 이미 바뀐 split으로
        # 쏠리는 왜곡이 생긴다(실측 데이터로 발견, P1_테스트결과서_ITSubClassification.md
        # 4절 참고) — 단일/복합을 뒤섞인 순서 그대로 처리해야 라벨별 목표 비율이
        # 유지된다.
        record = candidates[0]
        pool.remove(record)

        other_labels = [label for label in record.sub_categories if label != rarest_label]
        best_split = 0
        best_key: tuple[float, float, int] | None = None
        for idx in range(len(ratios)):
            # 1순위: 이 레코드를 뽑은 이유인 "가장 희소한 라벨"의 남은 목표치 —
            # 다른 라벨의 여유가 아무리 커도 희소 라벨의 배분을 우선 보호해야 한다
            # (그렇지 않으면 여러 라벨에 걸친 레코드가 특정 split으로 계속 쏠릴 수 있음).
            rarest_score = remaining_desired[rarest_label][idx]
            # 2순위: 나머지 라벨들의 남은 목표치 합(동률 시 참고용)
            other_score = sum(remaining_desired[label][idx] for label in other_labels)
            key = (rarest_score, other_score, -assigned_count[idx])
            if best_key is None or key > best_key:
                best_key = key
                best_split = idx

        assigned[best_split].append(record)
        assigned_count[best_split] += 1
        for label in record.sub_categories:
            remaining_desired[label][best_split] -= 1

    return assigned[0], assigned[1], assigned[2]
