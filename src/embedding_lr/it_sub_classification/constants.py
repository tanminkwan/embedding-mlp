"""2차(IT 세부 분류) 전용 도메인 상수 — Architecture_Design.md 9.2절 참고.
1차 constants.py는 5-class 단일 라벨 전제라 그대로 add할 수 없어 신규 정의한다."""

SUB_LABELS = ["DBA", "DEVOPS", "MIDDLEWARE", "NETWORK", "OS"]
FIELD_SUB_CATEGORY = "세부카테고리"

MIN_LABEL_POSITIVE_COUNT = 200
MIN_COMBO_POSITIVE_COUNT = 30
