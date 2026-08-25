# embedding-mlp

기 구축된 임베딩 서비스(**AIPro+**, BGE-M3 + Qdrant, `localhost:28000`)를 앞단에 두고,
2단계 캐스케이드로 실시간 쿼리를 분류하는 파이프라인.

1. **1차 분류(기존 유지)**: `embedding-lr`의 Logistic Regression 5-class 분류기를
   그대로 사용해 `IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY`를 판정한다. 이 단계는
   수정하지 않는다.
2. **2차 분류(신규)**: 1차 결과가 `IT`인 건에 대해서만, 작은 MLP 분류기가
   `dba`/`devops`/`os`/`network`/`middleware`/`etc` 6개 세부 카테고리로 재분류한다.
   `NON_IT`(1차가 `IT`가 아닌 경우)는 2차 분류를 타지 않는다. **2차는 멀티라벨
   분류**다 — 한 쿼리가 예: `DBA`이면서 동시에 `MIDDLEWARE`인 것처럼 6개 라벨 중
   **1개 이상**을 동시에 가질 수 있다(상호 배타적이지 않음).

자세한 배경은 [docs/Scope_Definition.md](docs/Scope_Definition.md) 참고(원본 5-class
스코프이며, 2차 MLP 분류 관련 요구사항/설계 문서는 `mlp-phase0`부터 순차 추가 예정).

## 분류 대상

### 1차: LR 5-class (기존, 변경 없음)

| 라벨 | 설명 | 최종 판정 |
|---|---|---|
| `IT` | IT 기술 질의 (2차 세부 분류 대상) | **IT** |
| `DAILY` | 일상 대화 | NON_IT |
| `KNOWLEDGE` | 일반 지식/교양 | NON_IT |
| `CREATIVE` | 창작/엔터테인먼트 | NON_IT |
| `ANOMALY` | 무의미 입력 | NON_IT |

### 2차: MLP 멀티라벨(신규, 1차가 `IT`인 건만 대상)

| 라벨 | 설명 |
|---|---|
| `DBA` | 데이터베이스 관리 관련 기술 질의 |
| `DEVOPS` | CI/CD, 배포, 운영 자동화 관련 기술 질의 |
| `OS` | 운영체제 관련 기술 질의 |
| `NETWORK` | 네트워크 관련 기술 질의 |
| `MIDDLEWARE` | 미들웨어(WAS, 메시지 큐 등) 관련 기술 질의 |
| `ETC` | 그 외 IT 기술 질의 |

위 6개 라벨은 상호 배타적이지 않다 — 한 쿼리에 **복수 라벨이 동시에** 붙을 수 있다.
예: "Tomcat에서 Oracle 커넥션 풀이 자꾸 끊기는데 WAS 쪽 문제인지 DB 쪽 문제인지
모르겠다" 같은 질의는 `MIDDLEWARE`+`DBA`가 함께 정답이다.

`final_verdict`(IT/NON_IT 이진 판정) 로직은 1차 결과 기준으로 변경 없음. 2차 라벨은
`final_verdict`에 영향을 주지 않고, `IT`로 판정된 건의 세부 분류 정보(복수 가능)로만
추가된다.

## 분류 모델: 1차 LR(유지) + 2차 MLP(신규)

1차 Logistic Regression(scikit-learn, `LogisticRegression`)은 기존 그대로 사용한다.
2차 세부 분류기만 작은 MLP로 신규 도입한다. 입력은 1차와 동일한 BGE-M3 임베딩
벡터(1024차원), 출력은 위 6개 IT 세부 라벨(`K=6`)에 대한 **각각 독립적인** 확률이다
(멀티라벨이므로 6개 확률의 합이 1일 필요는 없음).

**왜 2차만 LR이 아니라 MLP인가**: 2차 학습 데이터는 한 레코드에 여러 라벨(예: `DBA`+
`MIDDLEWARE`)이 동시에 붙는 멀티라벨 문제라, `dba`/`devops`/`os`/`network`/`middleware`/
`etc` 사이의 경계가 1차(IT vs NON_IT, 상호 배타적 5-class)보다 훨씬 미묘하다. LR의
선형 결정 경계로는 이런 겹치는 라벨 조합을 표현하기 어려워, 은닉층 2개로 비선형
표현력을 확보한 MLP를 2차에 한해 도입한다.

```
입력(1024) → [W1: 1024×64] → ReLU → [W2: 64×64] → ReLU → [W3: 64×K] → sigmoid   (K=6)
```

- 마지막 활성화는 **sigmoid**(라벨별 독립 이진 판정, one-vs-rest)다 — softmax가 아니다.
  클래스 간 상호 배타성을 가정하지 않으므로, 각 라벨의 sigmoid 출력이 threshold(기본
  0.5, 검증셋 기준 튜닝 대상) 이상이면 해당 라벨을 함께 부착한다.
- 손실 함수는 라벨별 binary cross-entropy(멀티라벨 표준)를 사용한다.
- 은닉층 2개(64 유닛)로 구성된 소형 네트워크 — 대형 트랜스포머가 아니라 임베딩 위에
  얹는 얕은 분류 헤드 수준을 유지한다.
- 2차 MLP는 1차에서 `IT`로 판정된 데이터만으로 별도 학습·추론한다(1차 모델과 독립된
  아티팩트, 예: `model_it_sub_<ver>.pkl`).
- `K`는 라벨 수(현재 6)로, `constants.py`에 정의된 라벨 목록 크기를 그대로
  따른다(하드코딩 금지 원칙, CLAUDE.md 4절).
- 1차 분류기·파이프라인(Phase 1~5)은 수정하지 않고, 2차 MLP 분류기를 위한 요구사항/설계
  문서는 `mlp-phase0` 이후 브랜치에서 CLAUDE.md 3절 절차(요구사항정의서 → 설계서 →
  코드/테스트 → 테스트결과서)에 따라 문서화한다.

## 파이프라인

```
[Phase 1] 데이터 준비(포맷 변환)   data/<version>/role_01~09_*.csv(이미 확보된 원본) → role_01~09_*.jsonl
[Phase 1.5] 데이터 조합/분할       role_*.jsonl → data.jsonl → train/test/val.jsonl
[Phase 2] 임베딩 변환              train/test/val.jsonl → *_vectors.parquet (AIPro+ 호출)
[Phase 3] 모델 학습                *_vectors.parquet → model_<ver>.pkl (GridSearchCV)
[Phase 4] 검증                     val_vectors.parquet + model.pkl → eval_report.md/json
[Phase 5] 추론 서비스              FastAPI 상시 서비스, model_<ver>.pkl 로드 후 실시간 분류
```

Phase 1은 새 데이터를 만드는 단계가 아니다 — 질의·응답 내용은 이미 확보되어 있고(현재는
CSV), Phase 1 코드는 그 원본을 JSONL로 변환하는 일만 한다.

전체 구조는 [docs/Architecture_Design.md](docs/Architecture_Design.md) 참고.

## 추론 서비스 사용법

사전 조건: **Embedding Service(`localhost:8000`)만** 호스트에서 떠 있으면 된다. AIPro+는
런타임에 전혀 호출하지 않으므로(코드 전체에 AIPro 참조 없음) 꺼져 있어도 무방하다.
`models/model.pkl`(Phase 3 학습 산출물)이 존재해야 하고, `.env`에 `MODEL_DIR`/
`EMBEDDING_SERVER_*` 값이 채워져 있어야 한다. `AIPRO_*`는 실제로 쓰이진 않지만
`Settings`(pydantic) 필수 필드라 값 자체(더미도 가능)는 채워둬야 기동이 실패하지
않는다(`.env.example` 참고).

```bash
# 기동 (docker-compose.yml, network_mode: host로 호스트의 Embedding Service에 접근)
docker compose up --build -d

# 상태/로그 확인
docker compose ps
docker compose logs -f inference

# 헬스체크
curl http://localhost:8080/health
# {"status":"ok"}

# 종료
docker compose down
```

`POST /classify`는 `items` 리스트를 받아 순서 대응하는 `results` 리스트를 반환한다.
`query`만 분류에 쓰이고 `response`는 무시된다(값은 필요하되 빈 문자열 `""`로 넣어도 됨).

```bash
curl -X POST http://localhost:8080/classify \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {"query": "쿠버네티스 파드가 CrashLoopBackOff 상태인데 어떻게 확인하나요?", "response": ""},
      {"query": "오늘 점심 뭐 먹을지 고민되는데 추천해줘", "response": ""},
      {"query": "조선시대 왕 순서 알려줘", "response": ""},
      {"query": "재미있는 영화 추천해줘", "response": ""},
      {"query": "asdkfj alksjdf 123123", "response": ""}
    ]
  }'
```

응답 예시(`results[i]`는 `items[i]`에 순서로 대응):

```json
{
  "results": [
    {
      "predicted_category": "IT",
      "final_verdict": "IT",
      "probabilities": {"IT": 0.973, "DAILY": 0.003, "KNOWLEDGE": 0.008, "CREATIVE": 0.008, "ANOMALY": 0.008}
    }
  ]
}
```

- `predicted_category`: 5-class(`IT`/`DAILY`/`KNOWLEDGE`/`CREATIVE`/`ANOMALY`) 중 확률 최댓값
- `final_verdict`: `IT`/`NON_IT` 이진 판정(`IT`가 아니면 전부 `NON_IT`)
- `probabilities`: 클래스별 확률(합계 1.0)
- 임베딩 서버(`EMBEDDING_SERVER_BASE_URL`) 호출 실패 시 HTTP 503(`EmbeddingServerError`) 반환

## 작업 절차

이 프로젝트는 코드보다 문서가 먼저다. [CLAUDE.md](CLAUDE.md) 3절에 따라 각 Phase(및
공통 모듈)는 아래 순서를 반드시 지킨다.

1. 요구사항정의서 — 무엇을, 왜
2. 설계서 — 입력/출력, 인터페이스, 데이터 흐름
3. 코드 + 테스트 코드 (등급별 TDD 기준은 CLAUDE.md 2절)
4. 테스트 결과서 — 실행 결과, 등급별 커버리지

문서 파일명 규칙: `[P<phase>_]<DocType>_<Topic>.md` (CLAUDE.md 7절).

## 문서 목록 (`docs/`)

| 문서 | 내용 |
|---|---|
| [Scope_Definition.md](docs/Scope_Definition.md) | 프로젝트 최초 요구사항(스코프, 분류 로직, AIPro+ 연동, Phase 로드맵) |
| [Architecture_Design.md](docs/Architecture_Design.md) | 전체 시스템 설계 — 모듈 구조, 데이터 흐름, 기술 스택, Docker 구성. 1차(Phase 0~5) 완료 후 **프로젝트 전체의 단일 아키텍처 문서**로 통합됨(0절 참고) — 과거 Phase별 요구사항정의서/설계서/테스트결과서(`P0_*`~`P5_*`)는 폐기, 상세 이력은 아래 진행 상황 표와 git 이력 참고 |

2차(IT 세부 분류) 이후 신규 문서는 Phase 접두사 없이 `<DocType>_<Topic>.md`로 추가되며,
이 표에도 계속 추가한다([[CLAUDE.md]] 7절).

## 진행 상황

Phase 0(공통 모듈)~Phase 5(추론)까지 코드가 구현·테스트된 상태다(`embedding-lr` 기준
5개 Phase 전체 완료, 1차 LR 분류기는 수정하지 않고 그대로 유지). 이후 `mlp-phase0`
브랜치부터 1차 결과가 `IT`인 건에 한해 `dba`/`devops`/`os`/`network`/`middleware`/`etc`
6종으로 재분류하는 **2차 MLP 분류기**를 추가하는 것을 목표로 후속 작업을 진행 중이다
(`Architecture_Design.md` 9절 To-Be 아키텍처 참고). 프로젝트는 이 확장부터 **유지보수
단계**로 전환되어, Phase별 세부 요구사항정의서/설계서/테스트결과서는 더 이상 개별
문서로 만들지 않고 `Architecture_Design.md`(불변 골격)+신규 델타 문서 조합으로 관리한다
(`Architecture_Design.md` 0절).

| 영역 | 상태 | 비고 |
|---|---|---|
| Scope/요구사항 정의 | 완료 | Scope_Definition.md |
| 전체 아키텍처 설계 | 완료 | Architecture_Design.md |
| **Phase 0** 공통 모듈 설계 | 완료 | Architecture_Design.md 2절 |
| **Phase 0** 로깅 표준 설계 | 완료 | Architecture_Design.md 참고(JSON 구조화 로그, Loki/Grafana 미연동) |
| **Phase 0** 공통 모듈 코드+테스트(`config`/`constants`/`domain`/`exceptions`/`run_context`/`logging_config`) | 완료 | 23 tests passed, A+B 등급 커버리지 100%, Docker(`docker/Dockerfile.pipeline`) 내부에서 실행 |
| **Phase 0** 임베딩 캐싱 설계 변경 | 완료 | MD5 해시 기반 레코드 중복 판별 제거 → `source` 필드에 분류 라벨값 저장, 콜렉션을 `<version>_<train\|test\|validation>`로 분리. 도메인(`DOMAIN_NAME`, 프로젝트 고정 1개)·콜렉션 모두 사전 등록 후에만 지식 데이터 등록 가능(둘 다 이미 존재하면 재등록하지 않음) — Scope_Definition.md 2.1절/Architecture_Design.md 참고 |
| Phase 1 요구사항 정의 | 완료 | CSV→**JSONL** 포맷 전환 포함(CSV 이스케이프 사고 재발 원천 차단) |
| **Phase 1** 설계(데이터 준비/조합/분할) | 완료 | `DataRepository` Protocol(`CsvRepository`=읽기 전용, `JsonlRepository`) 뒤로 원본 형식을 추상화 — 형식이 바뀌어도 `dataset.combine`/`dataset.split`은 무수정. `SPLIT_RATIOS`(3:1:1)/`RANDOM_SEED`(42)/`RECORDS_PER_CLASS`(200) 상수 추가 |
| **Phase 1** 코드+테스트(`csv_repository`/`jsonl_repository`/`dataset.combine`/`dataset.split`/`cli.run_phase1`/`cli.run_phase1_5`) | 완료 | 신규 모듈 100% 커버리지, Docker 내부에서 실행 |
| Phase 1 데이터(v0.1 → v0.2) | 완료 | `data/v0.1_from.Claude-Cowork/role_03_network.csv`의 CSV 이스케이프 오류를 원본에서 직접 수정 → `cli.run_phase1`/`run_phase1_5`로 실제 변환·조합·분할 실행 → `data/v0.2/{role_01~09,data,train,test,val}.jsonl` 생성(1,000건, 클래스당 200건, train/test/val 120/40/40) |
| **Phase 2** 텍스트 전처리(`text_cleaner`) 설계+코드+테스트 | 완료 | 코드펜스 구분자 제거(본문 보존)/스택 트레이스 라인 제거/공백 정규화 3규칙, 순서 고정, 100% 커버리지 |
| **Phase 2** 임베딩 변환 설계+코드+테스트(`collection`/`aipro_client`/`embedding_server_client`/`registration`/`knowledge_writer`/`pipeline`/`cli.run_phase2`) | 완료 | AIPro+ 도메인/콜렉션 idempotent 등록 → `POST /api/rag/knowledge` 레코드별 개별 등록(bulk-upload 미사용) → `GET /api/rag/knowledge` 일괄 조회 → `*_vectors.parquet` 저장, 콜렉션 건수 일치 시 재등록 스킵, `embed()`는 어디서도 호출 안 함. Phase 2 범위 97%/프로젝트 전체 99% 커버리지, 98 tests passed — 실제 AIPro+는 호출하지 않고 respx/fake로 테스트 |
| **Phase 3** 모델 학습 설계+코드+테스트(`domain.models` 갱신/`training.trainer`/`training.persistence`/`cli.run_phase3`) | 완료 | `GridSearchCV`+`PredefinedSplit`(train=-1, test=0)으로 test set 기준 탐색, 최적 조합 선정 후 train set만으로 재학습, F1-macro→Accuracy tie-break, 모델(`.pkl`)/탐색이력(`.json`) 재실행 시 미덮어쓰기. Phase 3 범위 99%/프로젝트 전체 99% 커버리지, 121 tests passed(구현 중 scikit-learn 1.9 API 변화로 `multi_class` 인자 제거 + NaN tie-break 정렬 버그 수정) |
| **Phase 4** 검증 설계+코드+테스트(`domain.models`/`constants`/`training.persistence` 갱신/`evaluation.metrics`/`evaluation.report`/`cli.run_phase4`) | 완료 | `val_vectors.parquet`+`model.pkl`+`hyperparams.json`으로 5-class/이진 Accuracy·F1-macro·Confusion Matrix·Classification Report 산출, test-vs-validation gap 지표(임계값 설정 파일, 초과 시 warning) 추가, 목표 미달이어도 exit code는 항상 0(루프백은 사람이 리포트 보고 판단). Phase 4 범위 99%/프로젝트 전체 99% 커버리지, 150 tests passed(구현 중 일부 클래스만 학습된 모델에서 `probs_to_labels` KeyError 발견·수정). 실데이터(`data/v0.2/val_vectors.parquet`) 검증: 5-class Accuracy 98.5%/이진 Accuracy 100%/F1-macro 0.985, gap 0.01(무경고) — 목표 3종 모두 달성 |
| **Phase 5** 추론 설계+코드+테스트(`domain.interfaces`(`TextClassifier` 신규)/`domain.models`/`config` 갱신/`inference.{embedding_lr_classifier,predictor,api}`/`cli.run_inference_server`) | 완료 | 분류 방식을 `TextClassifier` Protocol 뒤로 추상화(임베딩+LR은 `EmbeddingLRTextClassifier` 어댑터 하나 — 추후 NLI/앙상블 교체 시 `cli.run_inference_server.py` 조립부만 변경). `POST /classify`(리스트 요청→리스트 응답, 순서 대응)/`GET /health`. 임베딩은 질의(query) 단독(실 AIPro+ 실험으로 검증, `response` 필드는 무시). 모델은 서비스 기동 시 1회 로드, 로드 실패 시 기동 자체 실패. Phase 5 범위 99%/프로젝트 전체 99% 커버리지, 168 tests passed(재작업 없음). 실서비스(Embedding Service `localhost:8000`) E2E: IT/DAILY/KNOWLEDGE/CREATIVE/ANOMALY 5종 질의 모두 정확히 분류, `response` 필드 무영향 확인. `Dockerfile.inference` 신규(`docker-compose.yml`은 후속 과제) |
| Docker: `Dockerfile.pipeline` 뼈대 | 완료 | Phase 0 공통 모듈 테스트 실행용. 호스트가 사내 프록시 경유 환경이면 `docker build --build-arg http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --build-arg no_proxy=$no_proxy`로 프록시를 넘겨야 `pip install`이 성공함(자동 상속 안 됨) |
| Docker: `Dockerfile.inference` | 완료 | Phase 5와 함께 완료(위 행 참고) |
| Docker: `docker-compose.yml` | 완료(추론 단독) | 추론 서비스(Phase 5) 단독 기동용 — `network_mode: host`로 호스트의 Embedding Service(`localhost:8000`)/AIPro+(`localhost:28000`)에 접근, `models/model.pkl`을 읽기 전용으로 마운트. Phase 1~4 배치 파이프라인 통합 오케스트레이션은 여전히 후속 과제 |
| Loki/Grafana 연동 | 미착수 | 향후 방침만 기록된 상태, 지금은 stdout 로그까지만 |

이 표는 작업이 진행될 때마다 갱신한다.
