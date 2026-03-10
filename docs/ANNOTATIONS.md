# Project Annotations

이 문서는 현재 코드베이스의 **파일 단위 역할**과 **함수/클래스 단위 책임**을 빠르게 파악하기 위한 주석(annotation) 문서입니다.

## 1) `src/models.py`

### 파일 역할
- 로컬 `transformers` 기반 모델 추론 래퍼(`Model`)와 원격 HTTP 호출 래퍼(`ModelAPI`)를 제공합니다.
- 실험에서 필요한 핵심 기능(다음 토큰 logits, 샘플링, `log p(x)`, 길이 보정 familiarity score, 앙상블 샘플링)을 한 곳에서 다룹니다.

### `_load_env_file(env_path)`
- `.env` 파일을 읽어 환경변수를 로드합니다.
- 이미 셋업된 환경변수는 덮어쓰지 않습니다.
- 형식: `KEY=VALUE`, 주석/빈 줄은 무시합니다.

### `_ForwardCache` (dataclass)
- `forward` 이후 상태를 저장하는 내부 캐시입니다.
- 필드:
  - `input_ids`: 직전 입력 토큰 ID
  - `attention_mask`: 직전 attention mask
  - `next_token_logits`: 직전 다음 토큰 logits
- `sample()`이 `forward()` 호출 결과를 이어받아 동작하도록 연결합니다.

### `Model.__init__(...)`
- Hugging Face 모델/토크나이저를 로드합니다.
- 로딩 정책:
  - `AutoTokenizer(..., padding_side="left")`
  - `AutoModelForCausalLM(..., torch_dtype="auto", trust_remote_code=True, attn_implementation=...)`
- 디바이스 자동 선택(cuda 우선) 및 `eval()` 모드 전환을 수행합니다.

### `Model.forward(text, generation_config=None)` / `__call__`
- 입력 텍스트(단일/배치)를 토크나이즈 후 모델 forward를 수행합니다.
- 마지막 위치의 logits(`[:, -1, :]`)을 반환합니다.
- 반환 logits는 캐시에 저장되어 이후 `sample()`의 시작점이 됩니다.

### `Model._sequence_log_prob_and_counts(...)`
- 내부 유틸리티로, teacher-forcing 방식의 시퀀스 확률 계산에 필요한 두 값을 반환합니다.
  - `sequence_log_probs`: 각 샘플의 `log p(x)` 총합
  - `token_counts`: 유효 토큰 개수(패딩 제외)
- `log_prob()`와 `familiarity_score()`가 공통 사용합니다.

### `Model.log_prob(text, generation_config=None)`
- 문자열(또는 배치 문자열)에 대한 `log p(x)`를 계산해 반환합니다.
- 출력 shape: `[batch_size]` 텐서.

### `Model.familiarity_score(text, generation_config=None, length_penalty=1.0)`
- 길이 차이에 의한 스케일 편향을 줄인 familiarity 점수를 계산합니다.
- 공식: `log p(x) / (token_count ** length_penalty)`
- `length_penalty=1.0`이면 평균 log-prob 형태와 유사한 정규화가 됩니다.

### `Model._get_eos_token_ids()`
- EOS 토큰 ID를 집합 형태로 반환하는 내부 유틸입니다.
- EOS가 단일 int 혹은 리스트인 경우를 모두 처리합니다.

### `Model.sample(tokens=1, generation_config=None)`
- 직전 `forward()` 캐시를 시작점으로 다음 토큰을 autoregressive하게 샘플링합니다.
- EOS가 나오면 해당 배치 샘플은 조기 종료합니다.
- 반환값: 배치 크기만큼의 생성 문자열 리스트 `List[str]`.

### `Model.ensembled_sample(texts, tokens=1, ensemble_scheme="sum", generation_config=None)`
- 여러 프롬프트의 다음 토큰 logits를 앙상블해 1개 토큰을 뽑는 과정을 반복합니다.
- 앙상블 방식 지원: `sum`(기본), `mean`, `max`.
- EOS 생성 시 조기 종료합니다.
- 반환값: 생성 문자열 1개(`str`).

### `ModelAPI.__init__(url=None, model_name=None, auth_token=None, timeout=60, env_path=".env")`
- 원격 API 클라이언트를 초기화합니다.
- 우선순위: 인자 값 > 환경변수(`.env` 로드 포함).
  - `MODEL_API_URL`
  - `MODEL_API_MODEL_NAME`
  - `MODEL_API_TOKEN`(선택)

### `ModelAPI._build_headers()`
- HTTP 헤더를 생성합니다.
- 항상 `Content-Type: application/json` 포함.
- 토큰이 있으면 `Authorization: Bearer <token>` 추가.

### `ModelAPI.forward(prompt, generation_config=None)` / `__call__`
- chat-completion 형태 payload를 구성해 `requests.post`로 호출합니다.
- HTTP 에러 시 예외를 발생시키고, 정상 응답은 JSON으로 반환합니다.

### `model`, `model_api`
- 각각 `Model`, `ModelAPI`의 lowercase alias 클래스입니다.

---

## 2) `src/prompt.py`

### 파일 역할
- 템플릿 메시지를 실제 데이터로 렌더링하여 chat-completion 입력으로 변환하는 경량 유틸리티를 제공합니다.

### `Prompt.__init__(messages)`
- 메시지 템플릿 리스트를 받아 내부에 저장합니다.
- 원본 변조를 방지하기 위해 deepcopy를 사용합니다.

### `Prompt.to_chat_completion(data)`
- 각 메시지의 `content`에 `.format(**data)`를 적용해 렌더링합니다.
- 반환값: `[{"role": ..., "content": ...}, ...]` 형태의 chat-completion 메시지 리스트.

---

## 3) `src/prompts/llm_judge.py`

### 파일 역할
- LLM-as-a-Judge 평가용 기본 프롬프트 템플릿을 제공합니다.

### `LLM_JUDGE_PROMPT`
- 한국어 채점 지시문 템플릿 문자열입니다.
- placeholder:
  - `{question}`
  - `{reference_answer}`
  - `{candidate_answer}`
- 출력 형식을 JSON(`score`, `reason`)으로 강제하도록 설계되어 있습니다.

### `LLM_JUDGE_MESSAGES`
- chat-completion 포맷(system + user)으로 바로 사용할 수 있는 메시지 템플릿입니다.
- `Prompt` 클래스와 함께 렌더링해 Judge 호출에 사용합니다.

---

## 4) `data/example.jsonl`

### 파일 역할
- 실험용 소규모 QA 데이터셋(샘플)입니다.
- 각 줄은 JSON object이며 스키마는 아래와 같습니다.
  - `{"question": <str>, "answer": <str>}`
- 태스크 구성:
  - 요약 10개
  - 교정 10개
  - 글쓰기 10개
  - 분류 10개

---

## 5) `.env.example`

### 파일 역할
- `ModelAPI` 실행에 필요한 환경변수 템플릿입니다.
- 포함 키:
  - `MODEL_API_URL`
  - `MODEL_API_MODEL_NAME`
  - `MODEL_API_TOKEN`(선택)

---

## 6) `.gitignore`

### 파일 역할
- 민감정보/캐시 파일을 버전관리에서 제외합니다.
- 현재 항목:
  - `.env`
  - `__pycache__/`
  - `*.pyc`



---

## 7) `src/utils/output.py`

### 파일 역할
- 실험 결과 산출물을 `output/{execute_time}` 규칙으로 저장하기 위한 유틸리티를 제공합니다.

### `make_output_dir(base="output", timestamp=None)`
- 실행 시각 기반 폴더를 만들고 `Path`를 반환합니다.

### `save_json(data, path)`
- 결과 딕셔너리/리스트를 UTF-8 JSON으로 저장합니다.

---

## 8) `experiments/prompt_familiarity.py`

### 파일 역할
- 질문의 familiarity score와 LLM Judge 성능 점수 간 상관관계를 측정하는 실험 스크립트입니다.

### 핵심 흐름
1. `data_path` JSONL 로드
2. `model_path`로 `Model` 초기화
3. 각 질문 배치에 대해 `familiarity_score`와 `sample` 생성 결과 획득
4. `judge_prompt_path`의 `LLM_JUDGE_MESSAGES` 템플릿을 로드/렌더링
5. `ModelAPI`로 judge 평가 점수 수집
6. Pearson/Spearman 상관계수 계산
7. scatter plot 및 결과 JSON 저장

### 출력
- `output/{execute_time}/sample_results.json`
- `output/{execute_time}/correlation.json`
- `output/{execute_time}/familiarity_vs_performance_scatter.png`
