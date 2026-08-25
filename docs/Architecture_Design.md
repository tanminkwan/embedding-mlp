# Application / System Architecture Design

[[Scope_Definition]]의 5-Phase 파이프라인(데이터 준비 → 임베딩 변환 → 모델 학습 → 검증 →
추론)을 실제 코드 구조로 구체화한 설계서. [[CLAUDE.md]]의 SOLID, workflow 친화, Docker
원칙을 반영한다.

## 0. 문서 체계 안내 (유지보수 전환, 신규)

1차(LR 5-class, Phase 0~5)는 완료되었고, 프로젝트는 2차(IT 세부 멀티라벨 MLP 확장,
[[Scope_Definition]] 2.2절)를 진행하는 **유지보수 단계**로 전환되었다. 이에 따라 문서
체계를 다음과 같이 정리한다.

- 1차 Phase별 요구사항정의서/설계서/테스트결과서(`P0_*`~`P5_*`, 총 20건)는 **폐기**한다.
  이 문서(`Architecture_Design.md`)를 프로젝트 전체(1차+2차)의 살아있는 단일 아키텍처
  문서로 통합 관리하며, 이번 갱신(To-Be)에서 1차 절(1~8절)의 내용은 실제 구현과 다르지
  않도록 최신화하고, 2차 확장분은 9절에 추가한다.
- 1차 완료 당시의 실행 이력(테스트 건수·등급별 커버리지·실측 검증 지표 등)은
  `README.md` "진행 현황" 표에 보존되어 있으므로 유실되지 않는다 — 상세 경위가 필요하면
  git 커밋 이력(`P0_*`~`P5_*` 삭제 이전 커밋)을 참고한다.
- 1차 Phase별 문서가 폐기되어 물리적 충돌이 없으므로, **Phase 번호(P0~P5)는 2차 신규
  산출물에도 그대로 재사용**한다 — [[Scope_Definition]] 7절 로드맵이 이미 Phase 0~5 각
  행에 1차(불변)/2차(신규 추가) 작업을 나란히 배치해 두었으므로, 문서 번호도 그 Phase에
  맞춘다(예: 2차 데이터 준비 요구사항정의서 → `P1_요구사항정의서_ITSubClassification.md`,
  [[CLAUDE.md]] 7절 "유지보수 단계 예외" 참고). 다만 내용은 **1차와 공통인 불변 부분을
  재서술하지 않고 이 문서를 참조**하며 **변경·추가되는 부분만** 다룬다.

## 1. 아키텍처 개요

배치 파이프라인(Phase 1~4)과 상시 구동 추론 서비스(Phase 5)를 분리한다. 각 Phase는
독립 CLI 진입점으로, 파일(디스크) 입출력을 통해서만 연동한다 — 함수 직접 호출로 체이닝
하지 않는다. 이는 향후 워크플로우 도구(Airflow류) 이식과, 실패한 Phase만 재실행하는
것을 가능하게 한다.

```
[Phase 1] 데이터 준비(포맷 변환)   data/<version>/role_01~09_*.csv(이미 확보된 원본) → role_01~09_*.jsonl
[Phase 1.5] 데이터 조합/분할       role_*.jsonl → data.jsonl → train/test/val.jsonl
[Phase 2] 임베딩 변환              train/test/val.jsonl → *_vectors.parquet (AIPro+ 등록+조회, 독립 Embedding Service 미사용)
[Phase 3] 모델 학습                *_vectors.parquet → model_<ver>.pkl (GridSearchCV)
[Phase 4] 검증                     val_vectors.parquet + model.pkl → eval_report.md/json
[Phase 5] 추론 서비스              FastAPI 상시 서비스, model_<ver>.pkl 로드 후 실시간 분류
```

Phase 1은 **새 데이터를 만드는 단계가 아니다** — 질의·응답 내용 자체는 이미 확보되어
있고(현재는 CSV), Phase 1의 코드는 그 원본을 학습 파이프라인이 쓰는 JSONL로 변환하는
일만 한다. 각 화살표는 "파일 경로"이며, 다음 Phase의 CLI는 이 경로를 `--input` 인자로
받는다. 상류 원본(`role_*.jsonl`)을 하류 결과가 절대 덮어쓰지 않는다(과거 `role_03_network.csv`의
CSV 이스케이프 오류로 인한 레코드 손실 사고 재발 방지 — role → data → train/test/val
순서만 허용).

## 2. 모듈 구조 (SOLID — SRP/DIP 중심)

```
src/embedding_lr/
├── config.py              # .env 로딩 (pydantic-settings)
├── constants.py           # 고정 도메인 상수: 5개 클래스 라벨, 데이터 split 종류(train/test/validation), 임베딩 차원(1024)
├── domain/
│   ├── models.py          # QueryRecord, KnowledgeRecord, KnowledgeItem, PredictionResult (dataclass/pydantic)
│   └── interfaces.py      # Protocol: EmbeddingClient(독립 Embedding Service, Phase 5 전용), VectorStore(AIPro+, Phase 2 전용), Classifier, DataRepository
├── preprocessing/
│   └── text_cleaner.py    # 코드펜스 구분자 제거 + 스택 트레이스 라인 제거 + 공백 정규화(순서 고정) — Phase2와 추론에서 공유
├── data_generation/       # Phase 1 — 이미 확보된 원본(현재 CSV)을 JSONL로 변환
│   ├── csv_repository.py    # `DataRepository` 구현체 — 레거시 CSV 읽기 전용(save는 미지원, CSV로는 내보내지 않음)
│   └── jsonl_repository.py  # `DataRepository` 구현체(JSONL) — 지금은 이 형식뿐이지만 형식이 바뀌면 이 구현체만 교체
├── dataset/                # Phase 1.5 — `list[QueryRecord]` 위에서만 동작, 파일 형식을 모른다(DIP)
│   ├── combine.py          # role 9개 `list[QueryRecord]` → 재조합, 클래스당 200건 검증
│   └── split.py            # `list[QueryRecord]` → 클래스별 3:1:1 stratified 분할 (seed 고정)
├── embedding/               # Phase 2(학습, AIPro+) + Phase 5 추론용 클라이언트(독립 Embedding Service)
│   ├── aipro_client.py      # VectorStore 구현체 — AIPro+ API(localhost:28000) HTTP 호출. get_knowledge()(GET /api/rag/knowledge, 임베딩 포함 조회). Phase 2 전용
│   ├── embedding_server_client.py  # EmbeddingClient 구현체 — AIPro+와 무관한 독립 Embedding Service(localhost:8000) HTTP 호출. embed()(POST /embed). Phase 5 추론 전용
│   ├── collection.py        # 순수 로직 — version+split(경로에서 자동 추출) → 콜렉션명 `<version>_<train|test|validation>` 생성 규칙 (외부 의존성 없음). AIPro+ collection_name 패턴(`^[a-zA-Z0-9_-]+$`, 점 불가)에 맞춰 version 문자열의 `.`을 `_`로 치환(예: v0.2 → v0_2)
│   ├── registration.py      # AIPro+ 사전 등록 보장(HTTP 호출) — ensure_domain(DOMAIN_NAME, 최초 1회) + ensure_collection(collection.collection_name(version, split)), 둘 다 이미 존재하면 무시(idempotent)
│   ├── knowledge_writer.py  # category(라벨)를 source 필드로 매핑해 AIPro+ POST /api/rag/knowledge 적재(content 기반, AIPro+가 내부에서 임베딩 계산) — 레코드 단위 중복 판별 없음, 콜렉션 전체 재등록
│   └── pipeline.py          # jsonl → registration.ensure_domain/ensure_collection → aipro_client.get_knowledge()로 콜렉션 기존 건수 확인 → 건수 일치 시 재등록 스킵, 불일치 시 text_cleaner → knowledge_writer(등록) → get_knowledge()(재조회) → parquet 저장. embed()는 호출하지 않음
├── training/                # Phase 3
│   ├── trainer.py           # Classifier 구현체 — sklearn LogisticRegression + GridSearchCV
│   └── persistence.py       # joblib save/load, 버전 관리(model_<ver>.pkl)
├── evaluation/               # Phase 4
│   ├── metrics.py            # accuracy/F1/confusion matrix, IT vs NON_IT 집계
│   └── report.py             # 테스트 결과서 생성 (md/json)
├── inference/
│   ├── predictor.py          # 모델 + EmbeddingClient(embedding_server_client) 조합, predict_proba
│   └── api.py                # FastAPI: POST /classify
└── cli/                      # 워크플로우 트리거 경계 — Phase별 독립 실행 진입점
    ├── run_phase1.py, run_phase1_5.py ... run_phase4.py
    └── run_inference_server.py
```

**DIP 적용 지점**: `embedding/pipeline.py`는 `VectorStore` Protocol(AIPro+)에, `inference/predictor.py`는
`EmbeddingClient` Protocol(독립 Embedding Service)에, `training/trainer.py`는 `Classifier`
Protocol에만 의존한다 — 서로 다른 두 외부 서비스(AIPro+/Embedding Service)를 각각
다른 Protocol로 분리했으므로(ISP), 한쪽 서비스를 교체해도 다른 경로는 영향받지 않는다.
LogisticRegression을 다른 분류기로 바꿔도 파이프라인 로직은 수정하지 않는다. 테스트에서는 이 Protocol을 가짜(fake) 구현으로 교체해 TDD를
수행한다. `dataset/combine.py`·`dataset/split.py`도 같은 원칙을 따른다 — 파일이 아니라
`DataRepository` Protocol이 반환한 `list[QueryRecord]` 위에서만 동작하므로, 원본 데이터의
형식이 지금은 CSV(`data_generation/csv_repository.py`, 읽기 전용)이고 저장은
JSONL(`data_generation/jsonl_repository.py`)로 하지만, 나중에 원본 형식이 또 바뀌어도
(예: 다른 포맷의 원본, Parquet, DB) 이 두 모듈과 CLI 오케스트레이션 로직은 수정하지
않고 `DataRepository` 구현체만 추가/교체하면 된다.

## 3. Workflow 친화 규약 (모든 Phase 공통)

각 `cli/run_phaseN.py`는 다음 계약을 지킨다.

| 항목 | 규약 |
|---|---|
| Trigger | `python -m embedding_lr.cli.run_phaseN --input <path> --output <path> [--config <path>]` |
| Input | 명시적 파일 경로 인자로만 받음(암묵적 상대경로 탐색 금지) |
| Output | 명시적 파일 경로 인자로만 씀. 지정 없으면 `<phase>_<timestamp>.ext`로 자동 버전링 — 기존 파일 덮어쓰기 없음 |
| 모니터링 | 시작/종료/실패를 `status/<phase>_<run_id>.json`에 기록 (started_at, ended_at, status, error) |
| 재시작 | 실패 시 해당 Phase만 동일 input으로 재실행 가능 — 이전 Phase 산출물은 그대로 보존됨 |

## 4. 데이터 흐름 상세

학습 경로(Phase 1~4)는 **파일**을 입출력으로 삼는 배치 흐름이고, 추론 경로(Phase 5)는
**REST API 요청 본문(request body)**을 입력으로, **분류 결과**를 출력으로 삼는 실시간
흐름이다. 두 경로는 `text_cleaner`(전처리)를 동일하게 공유하지만, 임베딩을 얻는 방식은
서로 다른 외부 서비스를 쓴다 — 학습 경로는 `aipro_client`(AIPro+, 지식 등록 후 일괄
조회)를, 추론 경로는 `embedding_server_client`(AIPro+와 무관한 독립 Embedding Service)를
쓴다([[Scope_Definition]] 2.1절). 학습 경로의 최종 산출물(`model_<ver>.pkl`)이 추론
경로로 넘어가는 유일한 접점이다.

Phase 2 진입 시 `registration.ensure_domain()`이 프로젝트 고정 도메인(`DOMAIN_NAME`
상수)이 AIPro+에 존재하는지 먼저 보장한다(최초 실행 시 1회 생성, 이후 실행은 존재
확인만 하고 통과 — idempotent). 그다음 입력 경로(`data/<version>/{train,test,val}.jsonl`)
에서 **`version`과 `split`(train/test/validation)을 자동 추출**해 `collection.py`의
순수 함수로 `<version>_<split>` 콜렉션명을 만들고, 이를 `registration.ensure_collection()`
이 AIPro+ `POST /api/collections`의 `collection_name`(시스템 내부 고유 ID)으로 등록한다
(`name`은 UI 표시용 별칭으로 별도 필드 — 예: 그대로 같은 문자열을 넣어도 무방). 도메인처럼
"하위"로 귀속되는 필드는 없고(`CollectionCreate`에 `domain_id`가 없음 — 실 API 확인,
2026-08-19), 도메인·콜렉션은 서로 독립적인 분류축이며 지식 데이터 등록 시
`domain_id`+`collection_name`을 함께 지정해 둘을 연결한다. `collection_name`은 AIPro+가
`^[a-zA-Z0-9_-]+$`만 허용(점 `.` 금지, 422로 검증)하므로 `collection.py`가 version
문자열의 `.`을 `_`로 치환한다(예: `v0.2` + `train` → `v0_2_train`). 이후 같은 버전·용도의
임베딩 upsert(`POST /api/rag/knowledge`)는 이 콜렉션에 귀속되어, 데이터 버전 또는
용도(train/test/validation)가 바뀌면 자동으로 별도 콜렉션으로 분리된다 — 하드코딩 없이
버전×용도와 콜렉션이 1:1로 매핑된다([[CLAUDE.md]] 4절). 도메인·콜렉션이 모두 사전
등록되어 있어야 `knowledge_writer.py`의 지식 데이터 적재가 성립한다 —
[[Scope_Definition]] 2.1절 "사전 등록 순서" 참고.

**의도(추적성)**: 항상 콜렉션 전체(train/test/validation 각각)를 한 번의 배치로 재적재하는
구조이므로, 레코드 단위 중복 판별(해시 비교 등)은 두지 않는다 — 재학습 시에는 해당
콜렉션을 전체 재등록(Upsert)한다. `knowledge_writer.py`가 `POST /api/rag/knowledge` 호출
시 `content`(정제된 텍스트)와 `source` 필드에 **분류 라벨값**(카테고리)을 실어 보내면
AIPro+가 내부에서 임베딩을 계산해 저장한다 — 이 프로젝트 코드는 임베딩을 직접 계산하지
않는다. `source`는 콜렉션 내에서도 라벨 단위로 데이터를 조회·추적할 수 있게 한다 —
[[Scope_Definition]] 8절 Golden Rule 3 "추적성 확보"와 동일한 목적.

**재실행 시 콜렉션 단위 재등록 스킵**: `ensure_collection()` 직후 `aipro_client.get_knowledge()`
(`GET /api/rag/knowledge`, `domain_id`+`collection`, `limit`=입력 split 레코드 수 이상)로
기존 등록 건수를 확인한다. 건수가 입력 JSONL 레코드 수와 일치하면 `text_cleaner`→
`knowledge_writer`(등록) 호출을 건너뛰고, 방금 조회한 결과의 `embedding`+`source`를 그대로
`*_vectors.parquet`으로 저장한다. 건수가 다르면(0건 포함) `text_cleaner`→`knowledge_writer`로
콜렉션 전체를 재등록한 뒤 다시 `get_knowledge()`로 조회해 parquet을 만든다 — 이 파이프라인은
어느 경로든 `embed()`를 호출하지 않는다(벡터는 항상 AIPro+가 계산·보관한 것을
`get_knowledge()`로 가져온다). 레코드 단위 비교는 하지 않으며, 판단 기준은 오직 콜렉션의
총 건수다([[Scope_Definition]] 2.1절 참고).

```mermaid
flowchart TD
    subgraph TRAIN["학습 경로 — 파일 기반 (Phase 1~4)"]
        direction TD
        A["data/&lt;version&gt;/role_01~09_*.csv<br/>(이미 확보된 원본)"] -->|"csv_repository.load + jsonl_repository.save"| B["role_01~09_*.jsonl"]
        B -->|"dataset.combine (재조합)"| C["data.jsonl"]
        C -->|"dataset.split (클래스별 3:1:1, seed 고정 분할)"| D["data/&lt;version&gt;/train.jsonl / test.jsonl / val.jsonl"]

        subgraph PHASE2["embedding.pipeline (Phase 2, split별 독립 실행)"]
            direction TD
            D --> U["registration.ensure_domain(DOMAIN_NAME)<br/>(최초 1회, idempotent)"]
            U -->|"AIPro+ POST /api/domains"| V["registration.ensure_collection(<br/>collection.collection_name(version, split))"]
            V -->|"AIPro+ POST /api/collections (name=version_split, domain=DOMAIN_NAME)"| W{"aipro_client.get_knowledge()<br/>건수 == 입력 레코드 수?"}
            W -->|"Yes (AIPro+ GET /api/rag/knowledge)"| H
            W -->|"No"| E["text_cleaner.clean_text()"]
            E --> G["knowledge_writer.py (content=정제 텍스트, source=label 매핑)"]
            G -->|"AIPro+ POST /api/rag/knowledge (collection=version_split, AIPro+가 내부 임베딩 계산)"| X["aipro_client.get_knowledge() (재조회)"]
            X -->|"AIPro+ GET /api/rag/knowledge"| H["train/test/val_vectors.parquet (1024D + label)"]
        end

        H -->|"training.trainer (GridSearchCV: C, solver, max_iter)"| I["model_&lt;ver&gt;.pkl + hyperparams.json"]
        I -->|"evaluation (val_vectors + model)"| J["eval_report_&lt;ver&gt;.md/json"]
        J -->|목표 달성 시 승격| K["model_&lt;ver&gt;.pkl (파일)"]
    end

    subgraph INFER["추론 경로 — REST API 기반 (Phase 5)"]
        direction TD
        M["POST /classify<br/>request body (질의+응답 JSON)"] -->|"inference.api"| N["text_cleaner.clean_text()"]
        N --> O["embedding_server_client.embed()"]
        O -->|"Embedding Service POST /embed (localhost:8000, AIPro+ 미사용)"| P["predictor.predict_proba()"]
        P --> Q["classification 결과<br/>(label + 클래스별 확률) → HTTP response"]
    end

    K -->|"inference.predictor 로드 (1회, 서비스 기동 시)"| P
```

- **입력이 파일이 아님**: 추론 경로는 `train/test/val.jsonl` 같은 파일을 거치지 않고, HTTP
  요청 본문(질의+응답 JSON)을 바로 `text_cleaner`에 넣는다.
- **출력이 파일이 아님**: 추론 결과도 파일로 쓰지 않고, `predict_proba()` 결과를 즉시
  HTTP 응답(label, 클래스별 확률)으로 반환한다.
- `model_<ver>.pkl`은 서비스 기동 시 1회 로드되며, 요청마다 다시 읽지 않는다(추론 경로의
  유일한 "파일" 접점).

## 5. 기술 스택 및 라이브러리

| 영역 | 선택 | 이유 |
|---|---|---|
| 언어 | Python 3.11+ | 기존 스코프(scikit-learn, AIPro+ API 클라이언트)와 정합 |
| 데이터 처리 | pandas | JSONL(`read_json(lines=True)`)/표 형태 데이터, 클래스별 stratified split에 용이 |
| 임베딩 벡터 저장 | pyarrow / parquet | 1024D float 배열을 JSONL보다 효율적으로 저장·로드 |
| 분류 모델 | scikit-learn (`LogisticRegression`, `GridSearchCV`, `train_test_split`, `accuracy_score`, `f1_score`, `confusion_matrix`) | Scope Definition에서 이미 확정된 선택 |
| 모델 직렬화 | joblib | sklearn 모델 저장 표준 |
| HTTP 클라이언트 | httpx | AIPro+ API 호출(`POST`/`GET /api/rag/knowledge`, `localhost:28000`, Phase 2 전용) + 독립 Embedding Service 호출(`POST /embed`, `localhost:8000`, Phase 5 전용), 타임아웃/재시도 설정 용이, 추론 서비스에서 async 재사용 가능 |
| 설정 관리 | pydantic-settings (+ `.env`) | 타입 안전한 환경변수 로딩, [[CLAUDE.md]] "하드코딩 금지" 규칙 구현체 |
| 스키마/검증 | pydantic | FastAPI 요청/응답 모델, `domain/models.py`의 데이터 클래스 |
| 추론 API | FastAPI + uvicorn | Scope Definition의 "실시간 쿼리 분류 파이프라인" 요구사항 충족 |
| 테스트 | pytest + unittest.mock, respx(HTTP 모킹) | TDD, AIPro+ 의존성 없는 단위 테스트 |
| 로깅 | 표준 logging (JSON 포맷) | Workflow 모니터링용 구조화 로그 |
| 컨테이너 | Docker, docker-compose | [[CLAUDE.md]] 6절 |

### 선택적(필요 시 도입)

| 후보 | 용도 | 도입 시점 |
|---|---|---|
| 커스텀 데이터 검증 함수(pandera 등) | Phase 1.5 조합 직후 JSONL 스키마/라벨 값 검증 — CSV 이스케이프 오류 같은 사고를 자동 감지 | Phase 1.5 구현 시 |
| mlflow (로컬 파일 백엔드) | 하이퍼파라미터 탐색 결과 실험 추적 | 재학습 반복이 많아질 때만, 과설계 방지를 위해 초기엔 `hyperparams.json` 기록으로 충분 |

## 6. Docker 구성

```
docker/
├── Dockerfile.pipeline     # Phase 1~4 배치 작업 공용 이미지 (pandas/sklearn/httpx/pydantic)
└── Dockerfile.inference    # Phase 5 상시 서비스 이미지 (+ fastapi/uvicorn)
```

- `Dockerfile.pipeline`: 단일 이미지, entrypoint는 `python -m embedding_lr.cli.run_phaseN`
  형태로 파라미터화. 배치 작업은 실행마다 컨테이너가 뜨고 끝나는 생명주기이므로, Phase별로
  이미지를 나누지 않고 실행 커맨드로만 구분한다.
- `Dockerfile.inference`: 상시 구동 서비스라는 별개의 생명주기이므로 별도 이미지로 분리.
- `docker-compose.yml`: `phase1`~`phase4`, `inference` 서비스 정의. 공통 `./data`, `./models`
  볼륨 마운트, `.env` 파일로 설정 주입.

## 7. 테스트 전략 (TDD 연계)

| 계층 | 대상 | 방식 |
|---|---|---|
| 단위 | `text_cleaner`, `collection`(콜렉션명 생성 규칙: `version_split`), `metrics` | 순수 함수, 외부 의존성 없음 |
| 단위(모킹) | `aipro_client`, `registration`, `knowledge_writer`, `predictor` | `EmbeddingClient`/`VectorStore` Protocol을 fake로 교체 또는 respx로 HTTP 모킹 |
| 통합 | `csv_repository`, `jsonl_repository`, `embedding.pipeline`, `training.trainer` | 소규모 fixture 데이터로 end-to-end 실행(파일 I/O는 `tmp_path`), 실제 AIPro+는 호출하지 않음 |
| E2E(수동/선택) | 전체 파이프라인 | 실제 AIPro+(`localhost:28000`) 대상, CI에는 포함하지 않음 |

## 8. 디렉터리 구조 요약

```
embedding-mlp/
├── .env.example
├── docker-compose.yml
├── docker/
│   ├── Dockerfile.pipeline
│   └── Dockerfile.inference
├── src/embedding_lr/        # 2절 참고
├── tests/{unit,integration}/
├── data/                     # 기존 유지
├── models/                   # model_<ver>.pkl, hyperparams.json
├── status/                   # Phase 실행 로그(JSON)
├── prompt/                   # 기존 유지
└── docs/                     # 기존 유지
```

## 9. 2차 확장 — IT 세부 분류(멀티라벨 MLP) To-Be 아키텍처 (신규)

[[Scope_Definition]] 2.2절·2.3절·3.4절·4.6절·7절에서 확정한 결정을 코드 구조 관점에서
정리한다. **1차(1~8절)는 이 확장으로 어떤 파일도 수정되지 않는다** — 8절 Golden Rule 4
"1차 불변 원칙". 상세 요구사항/설계는 이 절을 골격으로 후속
`P1_요구사항정의서_ITSubClassification.md`/`P1_설계서_ITSubClassification.md`(Phase
번호 재사용, 0절 참고)에서 구체화한다.

### 9.1 개요

1차 LR이 `IT`로 판정한 쿼리에 한해서만 2차 멀티라벨 MLP를 추가로 태우는 **캐스케이드**
구조다. 2차 라벨(`DBA`/`DEVOPS`/`OS`/`NETWORK`/`MIDDLEWARE`, 5종)은 상호 배타적이지
않으며 라벨별 독립 이진 판정(one-vs-rest, sigmoid+BCE)을 쓴다 — 1차의 단일 라벨
softmax 방식과 다른 별도 모델 아티팩트다. 당초 검토했던 6번째 라벨 `ETC`는 학습
라벨로 두지 않는다 — sigmoid 구조상 5개 라벨 확률이 모두 threshold 미만이면 그
자체로 "5종 어디에도 해당하지 않음"을 의미하므로, 별도 데이터 없이 **파생 상태**로
처리한다([[Scope_Definition]] 2.2절 v2.1).

**캐스케이드(게이팅)와 학습 데이터 구성은 별개 관심사다**: 위 캐스케이드는 **추론**
시에만 적용되는 규칙이다 — "1차가 IT라고 판정한 건만 2차를 태운다"는 2차 모델을
호출할지 말지의 게이팅 로직일 뿐, 2차 MLP를 **학습**할 데이터를 고르는 기준이 아니다.
2차 학습 데이터(`it_sub_data.jsonl`)의 `카테고리=IT`는 데이터 생성 시점에 부여된 정답
라벨이며, 이 레코드들을 실제로 1차 LR에 통과시켰을 때 IT로 판정되는지는 검증하지
않는다 — 1차도 ~98.5% 정확도(README 참고)인 이상 소수는 1차가 실제로는 `NON_IT`로
오분류할 수 있지만, 그런 레코드도 걸러내지 않고 그대로 2차 MLP 학습에 포함한다. 1차
모델 자체는 이 확장에서 재학습하지 않고 `embedding-lr`에서 완성된 것을 그대로
재사용한다([[Scope_Definition]] 4.6절).

### 9.2 모듈 구조 — 2차 전용 신규 패키지

2차 소유 산출물은 1차 모듈을 import/수정하지 않고 **신규 패키지 하나**로 격리한다
(2.3절 "2차가 소유하는 신규 모듈" 원칙). 1차의 `workflow.run_context`, `logging_config`,
`exceptions.EmbeddingLRError`는 신규 추가 없이 그대로 재사용한다(2.3절 검토 결과).
다만 `domain.models.QueryRecord`(단일 `category: str`)와 `domain.interfaces.DataRepository`
/`Classifier`는 **멀티라벨 타깃(레코드당 라벨 리스트)을 표현할 수 없어 재사용이
불가능**하다는 것이 `P1_설계서_ITSubClassification.md` 작성 중 확인됐다 — 2.3절 검토
당시 예상보다 재사용 가능 범위가 좁혀졌으며, 2차는 이 두 영역(도메인 모델/분류기
Protocol)에 한해 자체 타입을 정의한다.

```
src/embedding_lr/
├── (1차 모듈 전체 — 2절 기준, 무수정)
└── it_sub_classification/        # 2차 신규 패키지 — 1차 파일 무수정, 공용 추상화만 재사용
    ├── constants.py               # SUB_LABELS(5종), 임계값 등 2차 전용 상수(1차 constants.py에 add하지 않음)
    ├── domain/
    │   ├── models.py               # ITSubQueryRecord(query/response/sub_categories: list[str]) — 1차 QueryRecord 재사용 불가(단일 category 전제)
    │   └── interfaces.py           # ITSubDataRepository Protocol — 1차 DataRepository 재사용 불가(반환 타입이 list[QueryRecord] 고정)
    ├── data_generation/
    │   ├── it2_csv_repository.py   # ITSubDataRepository 구현체 — it2_*.csv(카테고리="LABEL1+LABEL2" 형식) 읽기 전용
    │   └── it_sub_jsonl_repository.py  # ITSubDataRepository 구현체(JSONL) — 읽기/쓰기
    ├── dataset/
    │   ├── combine.py              # 여러 소스 병합 + 라벨/조합 건수 검증
    │   └── split.py                # 반복 계층화(iterative stratification) 3:1:1 분할 — 1차 dataset/split.py와 별개(단일 라벨 전제 재사용 불가), 외부 라이브러리 없이 직접 구현(`P1_설계서_ITSubClassification.md` 3.1절)
    ├── training/
    │   └── trainer.py             # 멀티라벨 전용 Protocol 구현체(가칭 `MultiLabelClassifier` — 1차 `Classifier.fit(X, y: list[str])`는 라벨 리스트를 못 담아 재사용 불가, Phase 3 설계서에서 확정), 1차 training/trainer.py와 별개 아티팩트
    ├── evaluation/
    │   └── metrics.py             # Subset Accuracy/라벨별 P·R·F1(micro/macro)/Hamming Loss
    └── inference/
        └── classifier.py          # 1차가 IT로 판정한 건만 로드해 재분류, sub_categories(threshold 미만 5개 전부면 빈 리스트="미분류")/sub_probabilities 산출
```

- 재라벨링(기존 IT role 데이터 재검토) 산출물은 코드가 아니라 데이터/문서 산출물이므로
  이 패키지가 아니라 `data/<version>/it_sub_*.jsonl` + 검토서(`P1_검토서_ITSubRelabeling.md`류)로
  남는다.
- CLI는 1차와 동일한 워크플로우 규약(3절)을 따르는 신규 진입점을 추가한다(예:
  `cli/run_it_sub_train.py`, `cli/run_it_sub_eval.py` — 정확한 파일명은 설계서에서 확정),
  기존 `run_phaseN.py`는 무수정.

### 9.3 데이터 흐름 (To-Be)

```mermaid
flowchart TD
    subgraph TRAIN2["2차 학습 경로 — it_sub_classification (신규, 1차 파일 무수정)"]
        direction TD
        A2["role_01~05_*.jsonl(1차, 읽기 전용) 재라벨링<br/>+ 신규 단일/복합 라벨 데이터"] -->|"재조합"| B2["it_sub_data.jsonl"]
        B2 -->|"it_sub_classification.dataset.split (반복 계층화 3:1:1)"| C2["it_sub_train/test/val.jsonl"]
        C2 -->|"embedding.pipeline 재사용(별도 콜렉션)"| D2["it_sub_*_vectors.parquet"]
        D2 -->|"it_sub_classification.training.trainer (멀티라벨 MLP)"| E2["model_it_sub_&lt;ver&gt;.pkl"]
        E2 -->|"it_sub_classification.evaluation"| F2["eval_report_it_sub_&lt;ver&gt;.md/json"]
    end

    subgraph INFER2["추론 경로 — 캐스케이드 (Phase 5 확장)"]
        direction TD
        G2["1차 predictor.predict_proba() 결과"] --> H2{"final_verdict == IT?"}
        H2 -->|"No"| I2["응답 반환 (sub_categories 없음)"]
        H2 -->|"Yes"| J2["it_sub_classification.inference.classifier<br/>(model_it_sub_&lt;ver&gt;.pkl, sigmoid+threshold)"]
        J2 --> K2["응답에 sub_categories(리스트)/sub_probabilities 추가"]
    end

    F2 -.->|목표 달성 시 승격| J2
```

- 임베딩 변환(Phase 2)은 코드 무수정으로 재사용하되, `it_sub_*.jsonl`을 입력으로 별도
  콜렉션(`<version>_it_sub_<split>`류, 정확한 명명은 설계서에서 확정)에 등록·조회한다.
- 추론 경로는 1차 `predictor.predict_proba()` 결과가 `IT`일 때만 2차 모델을 추가 로드해
  재분류한다 — 1차가 `IT`가 아니면 2차는 아예 실행되지 않는다.

### 9.4 Docker / 테스트 전략 확장

- Docker: `docker/Dockerfile.pipeline`(6절)를 그대로 재사용 — 2차 학습도 동일 배치
  이미지에서 신규 CLI 커맨드로만 구분한다. `Dockerfile.inference`는 2차 모델 로드
  로직이 추가되지만 이미지 자체는 그대로(의존성 변경 없으면 무수정).
- 테스트: `it_sub_classification/dataset/split.py`, `evaluation/metrics.py`는 등급 A(순수
  로직, TDD, 7절 표 기준)로 분류하고, `training/trainer.py`, `inference/classifier.py`는
  등급 B(오케스트레이션)로 분류한다 — [[CLAUDE.md]] 2절 등급 기준과 동일 원칙을 2차
  모듈에도 그대로 적용.
