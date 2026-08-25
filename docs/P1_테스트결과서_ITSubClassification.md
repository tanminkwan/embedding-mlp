# P1_테스트결과서_ITSubClassification — IT 세부 분류(2차) Phase 1 테스트 결과서

`P1_설계서_ITSubClassification.md`(설계) 구현의 테스트 실행 결과다. [[CLAUDE.md]] 3절
4단계 산출물. 실행은 [[CLAUDE.md]] 6절 원칙에 따라 Docker 컨테이너
(`docker/Dockerfile.pipeline`) 내부에서 수행했다.

## 1. 구현 산출물

설계서 2절 모듈 구조를 그대로 구현했다.

| 파일 | 등급 |
|---|---|
| `it_sub_classification/domain/models.py`(`ITSubQueryRecord`) | A |
| `it_sub_classification/domain/interfaces.py`(`ITSubDataRepository`) | — (Protocol 정의, 실행 로직 없음) |
| `it_sub_classification/dataset/combine.py`(`combine`/`label_counts`/`combo_counts`) | A |
| `it_sub_classification/dataset/split.py`(`iterative_stratified_split`) | A |
| `it_sub_classification/data_generation/it2_csv_repository.py`(`It2CsvRepository`) | B |
| `it_sub_classification/data_generation/it_sub_jsonl_repository.py`(`ITSubJsonlRepository`) | B |
| `cli/run_it_sub_phase1.py` | B |
| `cli/run_it_sub_phase1_5.py` | B |

## 2. 테스트 실행 결과

```
docker build -f docker/Dockerfile.pipeline -t embedding-mlp-pipeline:test \
  --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy .
docker run --rm embedding-mlp-pipeline:test python -m pytest --cov=embedding_lr --cov-report=term-missing -q \
  --ignore=tests/integration/test_api.py --ignore=tests/integration/test_run_inference_server.py
```

- **이번 Phase 1(2차) 신규 테스트: 35 passed**(단위 15 + 통합 20)
- **프로젝트 전체(제외 2건 — 이유는 4절): 197 passed**, 회귀(regression) 없음
- 제외한 2개 테스트 파일(`test_api.py`, `test_run_inference_server.py`)은 `fastapi`
  의존성이 `Dockerfile.pipeline` 이미지에 없어 원래부터 이 이미지에서 수집 불가능한
  기존 테스트다(Phase 5 추론 전용 extras, `Dockerfile.inference`/`inference` extras
  대상) — 이번 변경과 무관한 기존 조건이며 회귀가 아니다.

### 2.1 이번 Phase 1(2차) 모듈 커버리지

```
Name                                                                                Stmts   Miss  Cover   Missing
-----------------------------------------------------------------------------------------------------------------
src/embedding_lr/cli/run_it_sub_phase1.py                                              20      1    95%   40
src/embedding_lr/cli/run_it_sub_phase1_5.py                                            47      1    98%   96
src/embedding_lr/it_sub_classification/constants.py                                     4      0   100%
src/embedding_lr/it_sub_classification/data_generation/it2_csv_repository.py           21      0   100%
src/embedding_lr/it_sub_classification/data_generation/it_sub_jsonl_repository.py      33      0   100%
src/embedding_lr/it_sub_classification/dataset/combine.py                              27      0   100%
src/embedding_lr/it_sub_classification/dataset/split.py                                38      0   100%
src/embedding_lr/it_sub_classification/domain/interfaces.py                             3      3     0%   7-12
src/embedding_lr/it_sub_classification/domain/models.py                                17      0   100%
-----------------------------------------------------------------------------------------------------------------
```

| 등급 | 대상 모듈 | 목표(CLAUDE.md 2절) | 실측 |
|---|---|---|---|
| A | `domain/models.py`, `dataset/combine.py`, `dataset/split.py` | ≥90% | **100%** |
| B | `data_generation/*.py`, `cli/run_it_sub_phase1*.py` | ≥70% | **95~100%**(`__main__` 가드 각 1줄만 미달성 — 1차 CLI들과 동일한 패턴, 실행 진입점이라 테스트 대상 아님) |

`domain/interfaces.py`(`ITSubDataRepository` Protocol)는 `...`(구현부 없는 시그니처)만
있어 실행 커버리지 측정 대상이 아니다 — 1차 `domain/interfaces.py`도 동일 성격이나,
1차는 여러 구현체가 이미 존재해 간접적으로 100% 커버되는 차이가 있을 뿐 Protocol
자체의 성격은 같다.

### 2.2 프로젝트 전체 커버리지(발췌 — 회귀 확인용)

1차 모듈 전부 기존과 동일하게 100%(또는 기존에 이미 100% 미만이던 `embedding/aipro_client.py`
92%는 이번 변경과 무관, 기존 수치 유지)를 유지했다 — 이번 작업이 1차 파일을 한 줄도
수정하지 않았다는 것과 일치한다.

## 3. 테스트 항목 요약

| 파일 | 테스트 수 | 확인 내용 |
|---|---|---|
| `test_it_sub_domain_models.py` | 5 | 단일/복수 라벨 허용, 빈 리스트/미정의 라벨/중복 라벨 거부 |
| `test_it_sub_combine.py` | 5 | 병합, (질의,세부카테고리 조합) 중복 검출, 라벨별/조합별 건수 집계 |
| `test_it_sub_split.py` | 5 | 손실·중복 없는 전량 배정, 3:1:1 근사 비율, 조밀한 데이터셋에서 전 split 라벨 커버리지, 시드 재현성, 시드 변경 시 순서 변화 |
| `test_it2_csv_repository.py` | 6 | 단일/조합 라벨 분해, 파일명이 아닌 카테고리 필드값 신뢰, 미정의 라벨·컬럼 누락 시 에러, `save()` 미지원 |
| `test_it_sub_jsonl_repository.py` | 7 | 로드/저장 왕복 일치, `카테고리`="IT" 고정 기록, 빈 줄 스킵, JSON 파싱 실패·키 누락·라벨 검증 실패 시 에러, 기존 파일 존재 시 저장 거부 |
| `test_run_it_sub_phase1.py` | 3 | CSV→JSONL 변환, 실행 상태 파일 기록, 출력 파일 존재 시 실패 |
| `test_run_it_sub_phase1_5.py` | 4 | 병합+분할 end-to-end, 상태 파일 기록, 라벨 희소로 일부 split 커버리지 누락 시에도 경고만 남기고 정상 종료, 출력 파일 존재 시 실패 |

합계 35 tests (단위 15 + 통합 20).

## 4. 재작업 내역

### 4.1 Docker 빌드 프록시 누락(경미)

최초 Docker 빌드 시 `--build-arg` 없이 실행해 `pip install`이 사내 프록시를 못 타고
네트워크 접근 실패로 빌드 자체가 실패했다(사용자 지적으로 발견) — README의 프록시
전달 관례([[Architecture_Design]] 6절)를 그대로 따라 재빌드해 해결. 코드 결함은 아니었음.

### 4.2 커버리지 보강

최초 구현 시 `it2_csv_repository`/`it_sub_jsonl_repository`의 에러 처리 분기(컬럼
누락, 라벨 검증 실패, 빈 줄 스킵)와 `run_it_sub_phase1_5`의 "일부 split 라벨 커버리지
누락" 경고 분기가 테스트로 다뤄지지 않아 커버리지가 90%대 초반이었다 — 해당 분기를
명시적으로 트리거하는 테스트를 추가해 100%(또는 `__main__` 가드 제외 전량)로 보강했다.

### 4.3 [중요] `iterative_stratified_split` 실데이터 검증 중 발견한 분할 비율 왜곡 버그

단위 테스트(작은 합성 데이터셋)는 전부 통과했지만, **실제 `it_sub_data.jsonl`
(850건)로 돌려보니 3:1:1(60:20:20)이어야 할 분할 비율이 71.5:14.2:14.2로 심하게
치우쳤다** — 단위 테스트만으로는 못 잡는 결함이었고, 실데이터 실행으로 검증하는
과정에서 발견했다.

- **원인**: `split.py`에서 "현재 남은 후보 중 가장 희소한 라벨"을 처리할 때, 그 라벨을
  가진 후보 레코드 중 **라벨 수가 적은(단일 라벨) 레코드를 우선 배정**하도록 정렬한
  것이 문제였다. 이 때문에 특정 라벨의 단일 라벨 레코드가 그 라벨의 train 목표
  예산을 먼저 다 써버리고, 나중에 처리되는 복합 라벨 레코드는 이미 예산이 바뀐
  test/val로 쏠렸다 — 실측 결과 단일 라벨 레코드의 99.8%(478/479)가 train으로,
  복합 라벨 레코드는 test/val에 집중 배정됨을 확인했다.
- **수정**: 후보 선택 시 라벨 수 기준 우선순위 정렬을 제거하고, 셔플된 pool 순서
  그대로 첫 후보를 선택하도록 변경(`split.py` 주석 참고). split 선택 기준도
  "레코드가 가진 모든 라벨의 남은 목표치 합"에서 "이 레코드를 뽑은 이유인 가장
  희소한 라벨의 남은 목표치를 1순위, 나머지 라벨은 2순위 tie-break"로 정정해
  희소 라벨 보호가 다른 라벨 때문에 흐려지지 않게 했다.
- **수정 후 실측**: 전체 비율 58.0/21.1/20.9%(목표 60/20/20에 근접), 라벨별로도
  전부 근접(예: DEVOPS 144:48:48 = 정확히 60:20:20), 단일/복합 레코드도 세 split에
  고르게 분산됨(5절 표 참고).
- 단위 테스트(`test_it_sub_split.py`)는 수정 후에도 전부 통과하지만, 실측에서 드러난
  "단일/복합 레코드 분산" 속성 자체를 직접 검증하는 단위 테스트는 추가하지 않았다 —
  실데이터 규모(850건, 다양한 라벨 조합)에서만 뚜렷이 드러나는 문제라 작은 합성
  데이터로 안정적으로 재현하는 테스트를 만들기 어려웠고, 대신 5절의 실데이터 실행
  결과로 이 속성을 계속 확인하기로 한다.

## 5. 실데이터 실행 결과 (요구사항정의서 §7 완료 기준)

수정된 코드로 `data/v0.4/it2_*.csv`(15개, 650건) + `it_sub_relabel_role.jsonl`(200건)에
대해 `cli/run_it_sub_phase1.py`(15회) → `cli/run_it_sub_phase1_5.py`를 실제 실행했다.

| 항목 | 결과 |
|---|---|
| `it_sub_data.jsonl` | 850건, JSON 파싱 실패 0건, (질의,세부카테고리) 중복 0건 |
| 라벨별 양성 건수 | DBA 242 / DEVOPS 240 / MIDDLEWARE 242 / NETWORK 254 / OS 243 — 전부 목표(≥200) 초과 |
| `it_sub_train.jsonl` / `it_sub_test.jsonl` / `it_sub_val.jsonl` | 493 / 179 / 178건(58.0% / 21.1% / 20.9%, 목표 60/20/20에 근접) |
| Split 라벨 커버리지 | 5개 라벨 전부 train/test/val 각각 1건 이상(실측: 라벨별 대략 60/20/20 비율로 고르게 분포) |
| 재현성 | 동일 입력·시드(42, 기본값)로 재실행 시 바이트 단위로 동일한 출력 확인 |
| 1차 산출물 무변경 | `data/v0.2/role_01~09_*.jsonl`/`data.jsonl`/`train/test/val.jsonl` 무수정 확인 |

이로써 요구사항정의서 §7 완료 기준 중 남아있던 항목(`it_sub_data.jsonl` 생성, 반복
계층화 분할 실행, Split 라벨 커버리지 확인)이 모두 충족됐다.

## 6. 결론

설계서 §5(테스트 전략) 목표와 요구사항정의서 §7 완료 기준을 모두 충족했다. 다만
4.3절의 버그가 시사하듯, **작은 합성 데이터로 통과한 단위 테스트만으로는 실제
데이터 분포에서 드러나는 결함을 놓칠 수 있다** — 이후 유사한 배분/집계 로직을
추가할 때는 실데이터(또는 실데이터와 통계적으로 유사한 규모의 fixture) 실행 검증을
병행하는 것을 권장한다.
