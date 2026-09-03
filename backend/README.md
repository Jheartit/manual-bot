# 생명사 청약/배서 매뉴얼 AI 봇 — 파이프라인

## 구성 파일

| 파일 | 역할 |
|---|---|
| `ingest.py` | Google Drive 폴더 순회, 파일 목록/다운로드 |
| `parse.py` | PDF/이미지/xlsx → 텍스트 변환 (OCR 포함) |
| `embed_store.py` | 청킹, 임베딩, pgvector 저장 |
| `run_pipeline.py` | 위 세 단계를 묶어 전체 실행 (매일 1회 cron 권장) |
| `rag_query.py` | 질문 → 검색 → Claude 답변 생성 FastAPI 서버 |

## 준비 단계

1. **Google Cloud 서비스 계정 생성**
   - GCP 콘솔 → IAM → 서비스 계정 생성 → JSON 키 다운로드
   - 대상 Drive 폴더(청약/배서 매뉴얼)에 해당 서비스 계정 이메일을 "뷰어"로 공유

2. **Postgres + pgvector 준비**
   - `CREATE EXTENSION vector;` 가능한 Postgres 인스턴스 필요 (Supabase 무료 티어로도 가능)

3. **API 키 발급**
   - Voyage AI (임베딩): https://www.voyageai.com
   - Anthropic API (Claude): https://console.anthropic.com

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
3. 채팅창에 질문 입력 → 실제 벡터DB 검색 + Claude 답변이 돌아옴

## 배포 (박준희씨 혼자 또는 소수 사용 기준 최소 구성)

- **백엔드**: Railway, Render, Fly.io 같은 PaaS에 `rag_query.py`를 FastAPI 앱으로 배포 (Dockerfile 없이도 대부분 자동 감지됨)
- **DB**: Supabase 무료 티어(Postgres + pgvector 내장)를 쓰면 별도 DB 서버 구축 불필요
- **프론트엔드**: `manual_bot_ui_prototype.html`을 Netlify/Vercel에 정적 파일로 올리거나, 사내 인트라넷 서버에 올려두고 링크 공유
- **파이프라인 자동 실행**: `run_pipeline.py`를 위 PaaS의 cron job 기능(또는 GitHub Actions 스케줄)으로 매일 1회 실행되도록 등록

## 다음 개선 방향

- `run_pipeline.py`를 변경분(modified_time)만 재처리하도록 최적화
- 청킹을 섹션/조항 단위로 개선 (현재는 단순 슬라이딩 윈도우)
- 답변 캐싱 (자주 묻는 질문 대응)
- 인증/접근 제어 추가 (사내 인원만 접근 가능하도록)
