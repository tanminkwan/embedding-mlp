from embedding_lr.it_sub_classification.constants import SUB_LABELS
from embedding_lr.it_sub_classification.dataset.combine import label_counts
from embedding_lr.it_sub_classification.dataset.split import iterative_stratified_split
from embedding_lr.it_sub_classification.domain.models import ITSubQueryRecord


def _single_label_records(label: str, count: int, prefix: str) -> list[ITSubQueryRecord]:
    return [
        ITSubQueryRecord(query=f"{prefix}-{i}", response="r", sub_categories=[label])
        for i in range(count)
    ]


def _combo_records(labels: list[str], count: int, prefix: str) -> list[ITSubQueryRecord]:
    return [
        ITSubQueryRecord(query=f"{prefix}-{i}", response="r", sub_categories=list(labels))
        for i in range(count)
    ]


def _dataset() -> list[ITSubQueryRecord]:
    records = []
    for label in SUB_LABELS:
        records.extend(_single_label_records(label, 40, f"single-{label}"))
    records.extend(_combo_records(["DBA", "MIDDLEWARE"], 20, "combo-dba-mw"))
    records.extend(_combo_records(["OS", "NETWORK"], 20, "combo-os-net"))
    return records


class TestIterativeStratifiedSplit:
    def test_returns_three_groups_covering_all_records_exactly_once(self):
        dataset = _dataset()

        train, test, val = iterative_stratified_split(dataset)

        all_queries = [r.query for r in train + test + val]
        assert sorted(all_queries) == sorted(r.query for r in dataset)
        assert len(all_queries) == len(dataset)

    def test_approximately_follows_3_1_1_ratio(self):
        dataset = _dataset()

        train, test, val = iterative_stratified_split(dataset)

        total = len(dataset)
        assert abs(len(train) / total - 0.6) < 0.05
        assert abs(len(test) / total - 0.2) < 0.05
        assert abs(len(val) / total - 0.2) < 0.05

    def test_every_label_appears_in_every_split_for_dense_dataset(self):
        dataset = _dataset()

        train, test, val = iterative_stratified_split(dataset)

        for split_records in (train, test, val):
            counts = label_counts(split_records)
            for label in SUB_LABELS:
                assert counts[label] > 0, f"{label} missing from a split"

    def test_is_reproducible_with_same_seed(self):
        dataset = _dataset()

        result_a = iterative_stratified_split(dataset, seed=7)
        result_b = iterative_stratified_split(dataset, seed=7)

        for group_a, group_b in zip(result_a, result_b):
            assert [r.query for r in group_a] == [r.query for r in group_b]

    def test_different_seed_can_produce_different_order(self):
        dataset = _dataset()

        result_a = iterative_stratified_split(dataset, seed=1)
        result_b = iterative_stratified_split(dataset, seed=2)

        assert [r.query for r in result_a[0]] != [r.query for r in result_b[0]]
