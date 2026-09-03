# 생명사 청약·배서 매뉴얼 AI 봇

Google Drive의 생명사별 청약/배서 매뉴얼(PDF·이미지·xlsx)을 근거로 답변하고,
자료에 없는 내용은 웹 검색으로 보완하는 사내 업무 보조 봇.

## 폴더 구조

```
manual-bot/
├── backend/            # 수집·파싱·임베딩·RAG 질의응답 API
│   ├── ingest.py        # Google Drive 파일 수집
│   ├── parse.py         # PDF/이미지/xlsx → 텍스트 변환 (OCR 포함)
│   ├── embed_store.py   # 청킹, 임베딩, pgvector 저장
│   ├── run_pipeline.py  # 전체 파이프라인 실행 (수집→파싱→저장)
│   ├── rag_query.py     # 질의응답 FastAPI 서버
│   ├── requirements.txt
│   └── .env.example     # 환경변수 예시 (실제 .env는 커밋하지 않음)
└── frontend/
    └── index.html        # 챗봇 웹 UI
```

## 시작하기

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows는 .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # 값 채워넣기
python run_pipeline.py           # 최초 1회 전체 수집/임베딩
uvicorn rag_query:app --reload --port 8000
```

그다음 `frontend/index.html`을 브라우저로 열면 됩니다.

자세한 설정(서비스 계정 발급, pgvector 준비 등)은 `backend/` 안의 설명을 참고하세요.

## 주의

`.env`, `service_account.json` 등 비밀키 파일은 `.gitignore`에 포함되어 있어
커밋되지 않습니다. 실수로라도 절대 직접 `git add`하지 마세요.
