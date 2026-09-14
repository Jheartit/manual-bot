# 생명사 청약/배서 매뉴얼 AI 봇 — 파이프라인

## 구성 파일

| 파일 | 역할 |
|---|---|
| `ingest.py` | Google Drive 폴더 순회, 파일 목록/다운로드 |
| `parse.py` | PDF/이미지(tif 등 포함)/xlsx/pptx/docx/doc/zip/hwp → 텍스트 변환 (GPT-4o Vision OCR 포함) |
| `embed_store.py` | 청킹, 임베딩, pgvector 저장 |
| `run_pipeline.py` | 위 세 단계를 묶어 전체 실행 (매일 1회 cron 권장) |
| `rag_query.py` | 질문 → 검색 → GPT 답변 생성 FastAPI 서버 |

## 준비 단계

1. **Google Cloud 서비스 계정 생성**
   - GCP 콘솔 → IAM → 서비스 계정 생성 → JSON 키 다운로드
   - 청약/배서 매뉴얼 통합 루트 폴더에 해당 서비스 계정 이메일을 "뷰어"로 공유
   - 루트 폴더 구조는 아래와 같아야 함 (하위 폴더명에 "청약"/"배서"라는 단어가 포함되면 인식됨):
     ```
     루트 폴더 (DRIVE_ROOT_FOLDER_ID)
     ├── 생명사_청약매뉴얼(박준희)/
     │     ├── 삼성생명/파일1.pdf ...
     │     └── 한화생명/...
     └── 생명사_배서매뉴얼(박준희)/
           ├── 삼성생명/...
           └── ...
     ```
   - 같은 루트 아래에 다른 담당자의 청약/배서 폴더가 함께 있으면 `DRIVE_OWNER_TAG`에
     `박준희`처럼 폴더명에 포함된 문자열을 지정해 대상 폴더만 골라내야 함

2. **Postgres + pgvector 준비**
   - `CREATE EXTENSION vector;` 가능한 Postgres 인스턴스 필요 (Supabase 무료 티어로도 가능)

3. **API 키 발급**
   - Voyage AI (임베딩): https://www.voyageai.com
   - OpenAI API (GPT 답변 생성 + Vision OCR): https://platform.openai.com/api-keys
     (결제 수단 등록 필요, `web_search_preview` 도구를 쓰려면 Responses API 사용 가능한 계정이어야 함)

4. `.env.example`을 `.env`로 복사 후 값 채우기

## 실행 순서

```bash
pip install -r requirements.txt

# 1) 최초 1회 전체 수집 + 파싱 + 임베딩 저장
python run_pipeline.py

# 2) 질의응답 API 서버 실행
uvicorn rag_query:app --reload --port 8000
```

## API 사용 예시

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "삼성생명 암보험 청약 시 고지의무 조건이 뭐야?", "company_filter": "삼성생명"}'
```

응답:
```json
{
  "answer": "...",
  "sources": [{"company": "삼성생명", "doc_type": "청약", "file_name": "..."}]
}
```

## 프론트엔드 연결

`manual_bot_ui_prototype.html` 파일의 `API_BASE_URL` 값을 백엔드 서버 주소로 맞추면 바로 연결됩니다.

```js
// 로컬 테스트
const API_BASE_URL = 'http://localhost:8000';

// 배포 후
const API_BASE_URL = 'https://manual-bot.yourcompany.com';
```

로컬에서 테스트하는 순서:
1. `uvicorn rag_query:app --reload --port 8000` 로 백엔드 실행
2. `manual_bot_ui_prototype.html` 파일을 브라우저로 더블클릭해서 열기
3. 채팅창에 질문 입력 → 실제 벡터DB 검색 + GPT 답변이 돌아옴

## 배포 (박준희씨 혼자 또는 소수 사용 기준 최소 구성)

- **백엔드**: Railway, Render, Fly.io 같은 PaaS에 배포. `Procfile`(`web: uvicorn rag_query:app --host 0.0.0.0 --port $PORT`)이 있어 대부분 자동 인식됨
  - Root Directory를 `backend`로 지정해야 함
  - `.env`에 있는 값들을 그대로 플랫폼의 환경변수(Variables)로 등록
  - `GOOGLE_SERVICE_ACCOUNT_JSON`은 파일을 올리기 번거로우므로, `service_account.json` **파일 내용 전체**를 그 값으로 붙여넣으면 된다 (코드가 파일 경로/JSON 문자열 둘 다 인식함)
- **DB**: Supabase 무료 티어(Postgres + pgvector 내장)를 쓰면 별도 DB 서버 구축 불필요
- **프론트엔드**: `frontend/index.html`을 Netlify/Vercel에 정적 파일로 올리거나, 사내 인트라넷 서버에 올려두고 링크 공유. 배포 후 `API_BASE_URL`을 백엔드 배포 주소로 바꿔야 함
- **파이프라인 자동 실행**: `run_pipeline.py`를 위 PaaS의 cron job 기능(또는 GitHub Actions 스케줄)으로 매일 1회 실행되도록 등록
- **주의**: 현재 API는 인증이 없고 CORS도 전체 허용(`*`) 상태라, 배포 URL을 아는 사람은 누구나 사내 매뉴얼 내용에 접근할 수 있다. 접근 제어가 필요해지면 추가해야 함

## 증분 처리

`run_pipeline.py`는 Drive의 `modified_time`을 `manual_files` 테이블에 저장된 값과
비교해 신규/변경된 파일만 다운로드·파싱·재임베딩한다. Drive에서 삭제된 파일은
관련 청크/기록을 자동으로 정리한다.

```bash
python run_pipeline.py          # 변경분만 처리 (기본, 매일 cron 권장)
python run_pipeline.py --full   # 전체 강제 재처리
```

## 지원 파일 형식 관련 참고

- `.doc`(구버전 워드)는 macOS의 `textutil` 명령으로 변환한다. **Linux 서버(Railway/Render 등)에
  배포하면 `textutil`이 없어 .doc 파일 처리가 실패한다** — 배포 전 해당 파일들을 .docx로
  변환해두거나, LibreOffice 등 대체 변환 도구를 설치해야 한다.
- `.hwp`(한글과컴퓨터)는 공식 라이브러리 없이 OLE 구조를 직접 읽어 본문 텍스트만 뽑아내는
  최소 파서로 처리한다 (`parse.py`의 `parse_hwp`). 표/이미지/서식은 무시하고 문단 텍스트만
  가져오므로, 복잡한 레이아웃의 문서는 일부 내용이 유실될 수 있다.

## 다음 개선 방향

- 청킹을 섹션/조항 단위로 개선 (현재는 단순 슬라이딩 윈도우)
- 답변 캐싱 (자주 묻는 질문 대응)
- 인증/접근 제어 추가 (사내 인원만 접근 가능하도록)
- .hwp 파서가 표/이미지 등 본문 외 내용도 다루도록 개선
- .doc 지원을 macOS 전용(`textutil`)이 아닌 플랫폼 독립적인 방식으로 개선
