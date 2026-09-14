"""
4단계: 사용자 질문 → 벡터 검색 → GPT API 호출 (+ 부족하면 web_search로 보완)
FastAPI 서버로 감싸서 웹 UI에서 호출할 수 있게 구성.
"""
import os
import psycopg2
import voyageai
from openai import OpenAI
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 운영 시엔 실제 프론트엔드 도메인으로 제한
    allow_methods=["*"],
    allow_headers=["*"],
)

vo = voyageai.Client()
client = OpenAI(timeout=60.0, max_retries=2)  # OPENAI_API_KEY 환경변수 사용, 요청 행 방지
ANSWER_MODEL = "gpt-4o"

SYSTEM_PROMPT = """당신은 생명보험사 청약/배서 매뉴얼을 참고해 답변하는 업무 보조 봇입니다.

규칙:
1. 아래 제공되는 <검색된_자료> 안의 내용을 최우선 근거로 답변하세요.
2. 답변 시 반드시 어느 생명사, 어느 문서(파일명)에서 나온 내용인지 출처를 명시하세요.
3. <검색된_자료>에 질문에 대한 답이 없거나 부족하면, "자료 내에서 확인되지 않아 웹 검색으로 보완합니다"라고 밝힌 뒤 web_search 도구를 사용해 보완하세요.
4. 청약/배서 조건은 실무에 직접 영향을 주는 정보이므로, 답변 마지막에 "최종 확인은 원본 매뉴얼 또는 해당 생명사 담당자를 통해 진행해주세요."를 덧붙이세요.
5. 추측이나 확인되지 않은 정보를 자료나 검색 결과 없이 단정적으로 말하지 마세요."""


class QueryRequest(BaseModel):
    question: str
    company_filter: str | None = None  # 특정 생명사로 필터링 (선택)
    top_k: int = 8


def search_chunks(conn, question: str, company_filter: str | None, top_k: int):
    q_embedding = vo.embed([question], model="voyage-3", input_type="query").embeddings[0]

    with conn.cursor() as cur:
        if company_filter:
            cur.execute(
                """SELECT company, doc_type, file_name, content
                   FROM manual_chunks
                   WHERE company = %s
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (company_filter, q_embedding, top_k),
            )
        else:
            cur.execute(
                """SELECT company, doc_type, file_name, content
                   FROM manual_chunks
                   ORDER BY embedding <=> %s::vector
                   LIMIT %s""",
                (q_embedding, top_k),
            )
        return cur.fetchall()


def build_context(rows) -> str:
    parts = []
    for company, doc_type, file_name, content in rows:
        parts.append(f"[{company} / {doc_type} / {file_name}]\n{content}")
    return "\n\n---\n\n".join(parts)


def _looks_incomplete(answer: str) -> bool:
    """'웹 검색으로 보완합니다'라고 밝혀놓고 실제 검색 내용 없이 바로 끝내버리는
    부실한 답변이 가끔 나온다. 그런 경우를 감지해 재시도하기 위한 휴리스틱."""
    return "웹 검색" in answer and len(answer) < 250


def generate_answer(context: str, question: str, max_attempts: int = 2) -> str:
    """'웹 검색으로 보완합니다'라고 말해놓고 실제로는 web_search 도구를 호출하지
    않은 채 끝내버리는 경우가 있어, 그럴 땐 도구 호출을 강제해서 재시도한다."""
    answer = ""
    for attempt in range(max_attempts):
        force_search = attempt > 0
        resp = client.responses.create(
            model=ANSWER_MODEL,
            instructions=SYSTEM_PROMPT,
            tools=[{"type": "web_search_preview"}],
            tool_choice={"type": "web_search_preview"} if force_search else "auto",
            input=f"<검색된_자료>\n{context}\n</검색된_자료>\n\n질문: {question}",
        )
        answer = resp.output_text
        if not _looks_incomplete(answer):
            break
    return answer


@app.post("/query")
def query(req: QueryRequest):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    rows = search_chunks(conn, req.question, req.company_filter, req.top_k)
    conn.close()

    context = build_context(rows)
    answer = generate_answer(context, req.question)

    sources = list({(c, d, f) for c, d, f, _ in rows})

    return {
        "answer": answer,
        "sources": [{"company": c, "doc_type": d, "file_name": f} for c, d, f in sources],
    }


# 실행: uvicorn rag_query:app --reload --port 8000
