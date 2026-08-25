import pytest

from embedding_lr.exceptions import DataValidationError
from embedding_lr.it_sub_classification.data_generation.it2_csv_repository import It2CsvRepository


class TestIt2CsvRepositoryLoad:
    def test_splits_single_label_category(self, tmp_path):
        path = tmp_path / "it2_dba.csv"
        path.write_text('"질의","응답","카테고리"\n"q1","r1","DBA"\n', encoding="utf-8")

        records = It2CsvRepository().load(str(path))

        assert len(records) == 1
        assert records[0].query == "q1"
        assert records[0].sub_categories == ["DBA"]

    def test_splits_combo_label_category_on_plus(self, tmp_path):
        path = tmp_path / "it2_pair_middleware_dba.csv"
        path.write_text(
            '"질의","응답","카테고리"\n"q1","r1","MIDDLEWARE+DBA"\n', encoding="utf-8"
        )

        records = It2CsvRepository().load(str(path))

        assert records[0].sub_categories == ["MIDDLEWARE", "DBA"]

    def test_ignores_filename_and_trusts_category_field_only(self, tmp_path):
        # 파일명은 network+dba를 암시하지만 실제 카테고리 필드는 다른 조합 — 필드값이 우선.
        path = tmp_path / "it2_pair_network_dba.csv"
        path.write_text(
            '"질의","응답","카테고리"\n"q1","r1","OS+DEVOPS"\n', encoding="utf-8"
        )

        records = It2CsvRepository().load(str(path))

        assert records[0].sub_categories == ["OS", "DEVOPS"]

    def test_raises_on_unknown_label(self, tmp_path):
        path = tmp_path / "it2_dba.csv"
        path.write_text('"질의","응답","카테고리"\n"q1","r1","ETC"\n', encoding="utf-8")

        with pytest.raises(DataValidationError):
            It2CsvRepository().load(str(path))

    def test_raises_on_missing_column(self, tmp_path):
        path = tmp_path / "it2_dba.csv"
        path.write_text('"질의","응답"\n"q1","r1"\n', encoding="utf-8")

        with pytest.raises(DataValidationError, match="필수 컬럼"):
            It2CsvRepository().load(str(path))

    def test_save_not_implemented(self):
        with pytest.raises(NotImplementedError):
            It2CsvRepository().save([], "unused.csv")
