# P1_설계서_ITSubClassification — IT 세부 분류(2차) Phase 1 데이터 준비 설계서

`P1_요구사항정의서_ITSubClassification.md`(요구사항)의 구현 설계다. [[CLAUDE.md]] 3절
2단계 산출물이며, **1차와 공통인 불변 부분은 재서술하지 않고 [[Architecture_Design]]
(0절·9절)을 참조**한다. 이 문서는 Phase 1(데이터 준비) 범위의 모듈 구조·인터페이스·
데이터 흐름·알고리즘만 다룬다. 다음 산출물은 코드+테스트, 이어서
`P1_테스트결과서_ITSubClassification.md`(테스트결과서)다.

## 1. 현재 상태 요약 (입력 확정)

요구사항정의서 작성 이후 실제 데이터가 확보되어, §4.1의 "정확한 원본 파일명은 설계서에서
확정" 대상이 실제 파일로 확정됐다.

| 산출물 | 실제 경로 | 건수 |
|---|---|---|
| 재라벨링 결과([[P1_검토서_ITSubRelabeling]]) | `data/v0.4/it_sub_relabel_role.jsonl` | 200 |
| 신규 단일/복합 라벨 원본(CSV) | `data/v0.4/it2_{dba,devops,middleware,network,os}.csv`(단일 5개) + `data/v0.4/it2_pair_{dba_devops,middleware_dba,middleware_devops,middleware_network,middleware_os,network_dba,network_devops,os_dba,os_devops,os_network}.csv`(조합 10개) | 650 |

`it2_*.csv`의 `카테고리` 필드는 요구사항정의서가 가정한 `"IT"` 고정이 아니라 **라벨을
`+`로 직접 조인한 문자열**(예: `"MIDDLEWARE+DBA"`)이다 — 요구사항정의서 §4.1 스키마
가정과 다르므로 이 설계서에서 변환 규칙을 확정한다(2.3절).

## 2. 모듈 구조

[[Architecture_Design]] 9.2절의 `it_sub_classification/` 패키지를 아래와 같이
구체화한다. **1차 파일은 한 줄도 수정하지 않는다** — 아래 신규 모듈은 전부 이 패키지
안에만 추가된다.

```
src/embedding_lr/it_sub_classification/
├── constants.py               # SUB_LABELS = ["DBA","DEVOPS","MIDDLEWARE","NETWORK","OS"]
├── domain/
│   ├── models.py               # ITSubQueryRecord (신규, 2.1절)
│   └── interfaces.py           # ITSubDataRepository Protocol (신규, 2.2절)
├── data_generation/
│   ├── it2_csv_repository.py   # ITSubDataRepository 구현체 — it2_*.csv 읽기(카테고리 "+"분해), 읽기 전용
│   └── it_sub_jsonl_repository.py  # ITSubDataRepository 구현체 — it_sub 스키마 JSONL 읽기/쓰기
├── dataset/
│   ├── combine.py              # 여러 소스 병합 + 검증(2.4절)
│   └── split.py                # 반복 계층화 3:1:1 분할(3절 알고리즘)
└── (training/evaluation/inference — Phase 3~5 설계서에서 다룸, 이 문서 범위 아님)
```

### 2.1 `domain/models.py` — `ITSubQueryRecord` (신규)

1차 `domain.models.QueryRecord`는 `category: str | None`이 `CLASS_LABELS`(1차 5종)만
허용하도록 검증되어 있어, `세부카테고리`(멀티라벨 리스트)를 표현할 수 없다 — **1차
파일을 수정하지 않는다는 원칙(Golden Rule 4)에 따라 재사용하지 않고 2차 전용 모델을
새로 정의**한다.

```python
class ITSubQueryRecord(BaseModel):
    query: str
    response: str
    sub_categories: list[str]  # SUB_LABELS의 부분집합, 1개 이상, 중복 원소 금지

    @field_validator("sub_categories")
    @classmethod
    def _validate(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("sub_categories must have at least 1 label")
        if len(value) != len(set(value)):
            raise ValueError("sub_categories must not contain duplicates")
        unknown = set(value) - set(SUB_LABELS)
        if unknown:
            raise ValueError(f"unknown sub_categories: {unknown}")
        return value
```

`category`(1차 스키마의 `"IT"` 고정값) 필드는 이 모델에 두지 않는다 — 2차 대상은
항상 `IT`이므로 굳이 필드로 들고 다니지 않고, JSONL 입출력 시에만 `"카테고리": "IT"`를
고정 기록한다(요구사항정의서 §4.2 출력 스키마 그대로 유지, 도메인 모델은 2차 관심사인
`sub_categories`만 다룸).

### 2.2 `domain/interfaces.py` — `ITSubDataRepository` (신규)

1차 `domain.interfaces.DataRepository`도 반환 타입이 `list[QueryRecord]`로 고정돼
있어 재사용할 수 없다. 1차의 CSV→JSONL 추상화 패턴([[Architecture_Design]] 2절
DIP 적용 지점)과 동일한 취지로, 2차 전용 Protocol을 하나 더 둔다.

```python
class ITSubDataRepository(Protocol):
    def load(self, path: str) -> list[ITSubQueryRecord]: ...
    def save(self, records: list[ITSubQueryRecord], path: str) -> None: ...
```

- `It2CsvRepository`: `load`만 지원(원본 CSV는 읽기 전용, `save` 호출 시
  `NotImplementedError` — 1차 `CsvRepository`와 동일한 설계).
- `ITSubJsonlRepository`: `load`/`save` 둘 다 지원.

### 2.3 `it2_csv_repository.py` — 원본 CSV → `ITSubQueryRecord` 변환 규칙

`it2_*.csv`(질의/응답/카테고리, 카테고리는 `"LABEL"` 또는 `"LABEL1+LABEL2"`)를 다음
규칙으로 변환한다.

```python
def _row_to_record(row: dict) -> ITSubQueryRecord:
    labels = row["카테고리"].split("+")
    return ITSubQueryRecord(query=row["질의"], response=row["응답"], sub_categories=labels)
```

- `+` 분리 후 각 라벨은 `SUB_LABELS` 중 하나여야 한다(모델 검증기가 자동 확인).
- 파일명(`it2_pair_middleware_dba.csv` 등)은 참고용일 뿐 라벨 판단에 쓰지 않는다 —
  **`카테고리` 필드값만을 유일한 정답 소스로 취급**한다(파일명과 필드값이 어긋나는
  경우를 방지).

### 2.4 `dataset/combine.py` — 병합 + 검증

```python
def combine(sources: list[list[ITSubQueryRecord]]) -> list[ITSubQueryRecord]: ...
```

입력: [`it_sub_relabel_role.jsonl`을 `ITSubJsonlRepository.load()`한 결과] +
[`it2_*.csv` 15개 파일을 각각 `It2CsvRepository.load()`한 결과]를 리스트로 전달.

검증 규칙(요구사항정의서 §5 대응):

| 검증 항목 | 실패 시 동작 |
|---|---|
| (query, sorted(sub_categories)) 조합 중복 | `CombineValidationError` 발생, 중복 목록 출력 |
| `sub_categories` 원소가 `SUB_LABELS` 밖 | `ITSubQueryRecord` 생성 시점에 이미 `pydantic.ValidationError`로 차단 |
| 라벨별 최소 건수(≥200) 미달 | 경고 로그만 남기고 진행(차단 안 함 — 요구사항정의서 §4.5 반복 보강 절차가 후속 조치를 맡음) |
| 라벨 조합별 최소 건수(30~50) 미달 | 경고 로그만 남기고 진행(동일 사유) |

병합 결과를 `ITSubJsonlRepository.save()`로 `data/<version>/it_sub_data.jsonl`에
저장한다. 라벨별/조합별 건수 집계는 [[CLAUDE.md]] 2절 등급 A(순수 로직)로 별도 함수
(`label_counts(records) -> dict[str, int]`, `combo_counts(records) -> dict[str, int]`)로
분리해 단위 테스트한다.

## 3. `dataset/split.py` — 반복 계층화 3:1:1 분할

### 3.1 라이브러리 의존성 결정: 신규 의존성 추가 없음

`scikit-multilearn` 같은 전용 라이브러리 대신 **표준 라이브러리 + numpy만으로 직접
구현**한다 — 이유:
1. 데이터 규모가 작다(850건, 5라벨)여서 성능상 이점이 없다.
2. 알고리즘 자체가 검증 가능한 결정적 로직이라 등급 A(순수 로직, TDD)로 다루기
   적합하고, 외부 라이브러리의 블랙박스 동작보다 테스트로 직접 보증하는 쪽이
   추적성 원칙([[CLAUDE.md]] 4절)에 맞는다.
3. 새 의존성을 `pyproject.toml`에 추가하는 비용(버전 고정, Docker 이미지 재빌드) 대비
   이득이 작다.

### 3.2 알고리즘 (Sechidis et al. 2011 반복 계층화의 단순화 버전)

목표: 라벨별 3:1:1 비율을 각 split에서 최대한 보존하면서, 희소 라벨(조합)이 특정
split에 쏠리지 않게 한다.

```
1. 각 레코드가 가진 sub_categories 개수(라벨 카디널리티)를 계산한다.
2. 라벨별 목표 분배 카운트를 계산한다: 라벨 L의 전체 양성 건수 × (3/5, 1/5, 1/5)
   → train/test/val 각각의 목표치(반올림).
3. 아직 배정되지 않은 레코드 중, "현재 남은 후보들의 라벨 중 가장 희소한 라벨"을
   가장 적게 가진 레코드부터 우선 처리한다(희소 라벨을 먼저 배정해야 나중에 쏠림이
   덜 생김 — Sechidis 알고리즘의 핵심 아이디어).
4. 그 레코드가 가진 라벨들 중, "목표 대비 남은 배정 여유가 가장 큰 split"에 배정한다
   (동률이면 현재까지 배정된 전체 레코드 수가 가장 적은 split 우선 — 3:1:1 전체 비율도
   같이 맞추기 위함).
5. 모든 레코드가 배정될 때까지 반복한다.
6. 배정 순서는 **시드 고정 셔플**(입력 레코드 리스트를 `random.Random(RANDOM_SEED)`로
   먼저 섞은 뒤 위 순서로 처리)로 결정해, 동일 입력 재실행 시 항상 동일한 분할
   결과를 보장한다([[CLAUDE.md]] 4절 재현성, 1차 `RANDOM_SEED` 상수 재사용).
```

시그니처:

```python
def iterative_stratified_split(
    records: list[ITSubQueryRecord],
    ratios: tuple[int, int, int] = (3, 1, 1),
    seed: int = RANDOM_SEED,
) -> tuple[list[ITSubQueryRecord], list[ITSubQueryRecord], list[ITSubQueryRecord]]: ...
```

### 3.3 완료 기준 검증(요구사항정의서 §5 "Split 라벨 커버리지")

분할 후 `label_counts()`를 `it_sub_train/test/val` 각각에 적용해 5개 라벨이 모두
1건 이상 존재하는지 확인하는 assert를 `run_it_sub_combine_split.py`(4절) 실행 마지막
단계에 둔다 — 실패 시 CLI가 non-zero exit code로 종료한다(1차 Phase 1.5의 클래스당
200건 검증과 동일한 패턴).

## 4. CLI — Workflow 친화 규약 (1차와 동일 계약, [[Architecture_Design]] 3절)

1차의 "Phase 1(포맷 변환) → Phase 1.5(조합/분할)" 2단계 패턴을 그대로 따른다.

| CLI | 역할 | 입력 | 출력 |
|---|---|---|---|
| `cli/run_it_sub_phase1.py` | `it2_*.csv`(15개) → 정규화 JSONL 변환 | `data/<version>/it2_*.csv` | `data/<version>/it2_*.jsonl`(15개, `ITSubQueryRecord` 스키마) |
| `cli/run_it_sub_phase1_5.py` | 병합 + 반복 계층화 분할 | `data/<version>/it2_*.jsonl`(15개) + `data/<version>/it_sub_relabel_role.jsonl` | `data/<version>/it_sub_data.jsonl` + `it_sub_{train,test,val}.jsonl` |

`it_sub_relabel_role.jsonl`은 이미 `ITSubQueryRecord` 스키마이므로 별도 변환 단계 없이
Phase 1.5가 직접 입력으로 받는다. 두 CLI 모두 [[Architecture_Design]] 3절의
trigger/input/output/모니터링/재시작 규약을 그대로 따른다(`status/<phase>_<run_id>.json`
기록 등, `workflow.run_context` 재사용).

## 5. 테스트 전략 ([[CLAUDE.md]] 2절 등급 기준)

| 모듈 | 등급 | 근거 |
|---|---|---|
| `domain/models.py`(`ITSubQueryRecord`) | A | 순수 검증 로직, 입출력 결정적 |
| `dataset/combine.py`(`combine`, `label_counts`, `combo_counts`) | A | 순수 로직, 파일 I/O 없음(호출부가 Repository와 분리) |
| `dataset/split.py`(`iterative_stratified_split`) | A | 순수 로직, 동일 입력·시드 → 동일 출력 |
| `data_generation/it2_csv_repository.py`, `it_sub_jsonl_repository.py` | B | 파일 I/O 포함(오케스트레이션), 소규모 fixture로 통합 테스트 |
| `cli/run_it_sub_phase1.py`, `run_it_sub_phase1_5.py` | B | 오케스트레이션, `tmp_path` fixture로 end-to-end |

목표 커버리지: A등급 ≥90%, B등급 ≥70% ([[CLAUDE.md]] 2절 표 기준).

## 6. 산출물 요약

| 파일 | 설명 | 비고 |
|---|---|---|
| `data/<version>/it2_*.jsonl`(15개) | `it2_*.csv` 정규화 변환 결과 | 신규(Phase 1) |
| `data/<version>/it_sub_data.jsonl` | 전체 병합 결과(850건) | 신규(Phase 1.5) |
| `data/<version>/it_sub_train.jsonl` | 학습셋 | 신규(Phase 1.5) |
| `data/<version>/it_sub_test.jsonl` | 테스트셋 | 신규(Phase 1.5) |
| `data/<version>/it_sub_val.jsonl` | 검증셋 | 신규(Phase 1.5) |

`<version>`은 기존 관례상 `v0.4`를 그대로 쓴다(원본 CSV·재라벨링 결과가 이미
`data/v0.4/`에 있으므로, 1차의 `v0.2` 폴더가 원본과 조합·분할 결과를 함께 담은 것과
동일한 방식 — [[CLAUDE.md]] 5절 "입출력 보존"은 파일을 덮어쓰지 않는 것이 핵심이지
새 버전 폴더를 강제하지 않는다).

## 7. 리스크 및 참고사항

- `ITSubQueryRecord`/`ITSubDataRepository`가 1차 `QueryRecord`/`DataRepository`를
  재사용하지 못하는 것은 애초 [[Scope_Definition]] 2.3절이 예상한 "필드만 늘리면
  재사용 가능"의 범위를 벗어난 사례다 — 1차 `QueryRecord.category`가 단일 라벨
  전제([[CLAUDE.md]] 하드코딩 금지가 아니라 pydantic validator 제약)라서 리스트
  필드를 못 담기 때문이며, 같은 이유로 1차 `Classifier.fit(X, y: list[str])`도
  멀티라벨 타깃(`y`가 레코드당 라벨 집합)을 표현할 수 없다 — 이 점은 Phase 3
  설계서에서 별도로 `MultiLabelClassifier` 계열 Protocol을 2차 전용으로 새로
  정의해야 함을 미리 밝혀둔다(이 문서 범위 밖, Phase 3 설계서에서 확정).
- 반복 계층화 알고리즘은 라벨 조합이 극단적으로 희소한 경우(예: 특정 조합이 1~2건뿐)
  3:1:1 비율을 정확히 맞추지 못할 수 있다 — 이런 경우 해당 조합 전체가 한 split에
  몰릴 수 있으며, 이는 요구사항정의서 §4.5 반복 보강 절차(추가 생성)로 완화한다.
- `it2_*.csv`의 카테고리 인코딩(`LABEL1+LABEL2`)은 이번 데이터 생성에서 실제로
  쓰인 형식을 그대로 반영한 것이며, 향후 다른 형식의 원본이 추가되면
  `ITSubDataRepository` 구현체만 추가하면 된다(DIP, 1차와 동일 원칙).
