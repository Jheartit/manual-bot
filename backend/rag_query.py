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
ANSWER_MODEL = "gpt-4o-mini"  # gpt-4o는 기본 TPM 한도(30K)가 낮아 사용자가 늘면 429가 잦다

SYSTEM_PROMPT = """당신은 생명보험사 청약/배서 매뉴얼을 참고해 답변하는 업무 보조 봇입니다.

규칙:
1. 아래 제공되는 <드라이브_자료> 안의 내용을 최우선 근거로 답변하세요.
2. 답변 시 반드시 어느 생명사, 어느 문서(파일명)에서 나온 내용인지 출처를 명시하세요.
3. <드라이브_자료>에 질문에 대한 답이 없거나 부족하면, web_search 도구로 검색해 답변에 반영하고
   "자료 내에서 확인되지 않아 웹 검색으로 보완합니다"라는 문장만 답변 앞에 짧게 붙이세요.
   "찾아보겠습니다", "잠시만 기다려 주세요"처럼 지금부터 검색하겠다는 진행 상황 멘트는 절대 쓰지 마세요 —
   검색은 이미 끝난 상태로, 결과가 반영된 완성된 답변만 출력하세요.
4. 청약/배서 조건은 실무에 직접 영향을 주는 정보이므로, 답변 마지막에 "최종 확인은 원본 매뉴얼 또는 해당 생명사 담당자를 통해 진행해주세요."를 덧붙이세요.
5. 추측이나 확인되지 않은 정보를 자료나 검색 결과 없이 단정적으로 말하지 마세요.
6. web_search 결과가 질문과 실제로 무관하면(예: 보험과 관계없는 생활정보 등) 그 결과를 절대 답변에
   쓰지 마세요. 그런 경우엔 "자료 및 웹 검색에서 관련 정보를 찾지 못했습니다. 정확한 정보는 해당
   생명사 고객센터나 담당자를 통해 확인해주세요."라고만 답하세요."""


class QueryRequest(BaseModel):
    question: str
    company_filter: str | None = None  # 특정 생명사로 필터링 (선택)
    top_k: int = 8


MAX_CHUNKS_PER_FILE = 2  # 청크 수가 아주 많은 파일 하나가 검색 결과를 독점하지 못하게 제한


def search_chunks(conn, question: str, company_filter: str | None, top_k: int):
    """질문 임베딩과 코사인 거리순으로 문서를 찾되, 같은 파일에서 나온 청크가
    MAX_CHUNKS_PER_FILE개를 넘지 않도록 SQL 윈도우 함수로 먼저 걸러낸 뒤 top_k를 채운다.
    (콜센터 스크립트처럼 청크가 수만 개인 파일 하나가 임베딩 공간을 넓게 차지해버려서,
    후보를 아무리 넉넉히 가져와도 그 파일만 나오는 경우가 있어 후보 단계가 아니라
    SQL에서 파일당 순위를 매겨 원천적으로 다양성을 보장한다.)

    또한 HNSW 근사 인덱스는 이렇게 분포가 한쪽으로 쏠린 데이터에서 진짜 최근접 벡터를
    놓치는 경우가 있어(테스트로 확인됨), 이 정도 규모(1만 건대)에서는 인덱스를 끄고
    정확한 전수 비교로 검색한다."""
    q_embedding = vo.embed([question], model="voyage-3", input_type="query").embeddings[0]

    with conn.cursor() as cur:
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        # dist를 base 서브쿼리에서 한 번만 계산해 ranked에서 재사용 (두 번 계산하면
        # 눈에 띄게 느려짐 - 실측 3.2초 -> 1.9초로 단축)
        base_query = """
            SELECT company, doc_type, file_name, drive_file_id, content
            FROM (
                SELECT company, doc_type, file_name, drive_file_id, content, dist,
                       ROW_NUMBER() OVER (PARTITION BY file_name ORDER BY dist) AS rn
                FROM (
                    SELECT company, doc_type, file_name, drive_file_id, content,
                           embedding <=> %s::vector AS dist
                    FROM manual_chunks
                    {where_clause}
                ) base
            ) ranked
            WHERE rn <= %s
            ORDER BY dist
            LIMIT %s
        """
        if company_filter:
            query = base_query.format(where_clause="WHERE company = %s")
            params = (q_embedding, company_filter, MAX_CHUNKS_PER_FILE, top_k)
        else:
            query = base_query.format(where_clause="")
            params = (q_embedding, MAX_CHUNKS_PER_FILE, top_k)
        cur.execute(query, params)
        return cur.fetchall()


def build_context(rows) -> str:
    parts = []
    for company, doc_type, file_name, drive_file_id, content in rows:
        parts.append(f"[{company} / {doc_type} / {file_name}]\n{content}")
    return "\n\n---\n\n".join(parts)


def _looks_incomplete(answer: str) -> bool:
    """'웹 검색으로 보완합니다'라고 밝혀놓고 실제 검색 내용 없이 바로 끝내버리는
    부실한 답변이 가끔 나온다. 그런 경우를 감지해 재시도하기 위한 휴리스틱."""
    return "웹 검색" in answer and len(answer) < 250


def _final_message_text(resp) -> str:
    """도구를 쓰는 turn에서는 모델이 도구 호출 전에 미리 짧은 메시지를 하나 만들고,
    도구 호출 후 최종 메시지를 또 만드는 경우가 있다. resp.output_text는 이 둘을
    구분 없이 이어붙이기 때문에 중간 메시지 뒤에 최종 답변이 그대로 붙어버린다.
    실제로 보여줘야 할 건 output의 마지막 message뿐이다."""
    messages = [item for item in resp.output if item.type == "message"]
    if not messages:
        return resp.output_text
    last = messages[-1]
    return "".join(
        getattr(part, "text", "") for part in last.content
        if getattr(part, "type", None) == "output_text"
    )


def generate_answer(context: str, question: str, max_attempts: int = 3) -> str:
    """'웹 검색으로 보완합니다'라고 말해놓고 실제로는 web_search 도구를 호출하지
    않은 채 끝내버리는 경우가 있어 재시도한다.

    이전에는 재시도 시 tool_choice를 강제로 web_search_preview로 고정했는데,
    "쑥쑥이"처럼 애초에 실존하지 않거나 검색해도 안 나오는 상품명에 대해
    강제로 검색을 시키면 도구가 질문과 전혀 무관한 내용(예: 과일파리 퇴치법,
    목감기 치료법)을 가져와 그걸 그대로 답변에 반영해버리는 심각한 문제가 있었다.
    강제하지 않고 "auto"로만 재시도해, 모델이 검색해도 의미가 없다고 판단하면
    억지로 검색하지 않고 정직하게 "자료에 없다"고 답할 수 있게 한다."""
    answer = ""
    for _ in range(max_attempts):
        resp = client.responses.create(
            model=ANSWER_MODEL,
            instructions=SYSTEM_PROMPT,
            tools=[{"type": "web_search_preview"}],
            tool_choice="auto",
            input=f"<드라이브_자료>\n{context}\n</드라이브_자료>\n\n질문: {question}",
        )
        answer = _final_message_text(resp)
        if not _looks_incomplete(answer):
            break
    return answer


# "파일/양식 자체를 달라"는 취지의 질문인지 판단하는 키워드.
# 이런 질문이면 가장 유사도 높은 문서를 답변과 별도로 첨부파일 카드로 보여준다.
FILE_REQUEST_KEYWORDS = (
    "파일이 있", "파일 있", "양식이 있", "양식 있", "서식이 있", "서식 있",
    "예시 파일", "예시파일", "첨부해", "첨부 파일", "다운로드", "파일 좀",
    "파일을 보내", "파일 보내", "원본 파일", "샘플 파일",
)


def _looks_like_file_request(question: str) -> bool:
    return any(kw in question for kw in FILE_REQUEST_KEYWORDS)


@app.post("/query")
def query(req: QueryRequest):
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    rows = search_chunks(conn, req.question, req.company_filter, req.top_k)
    conn.close()

    context = build_context(rows)
    answer = generate_answer(context, req.question)

    sources = list({(c, d, f, fid) for c, d, f, fid, _ in rows})

    attachment = None
    if rows and _looks_like_file_request(req.question):
        # GPT가 답변 본문에서 실제로 인용한 파일명을 우선 찾는다 (검색 1위 파일과
        # GPT가 답변 근거로 실제 인용한 파일이 다를 수 있어, 그대로 rows[0]을 쓰면
        # 답변 내용과 첨부파일이 서로 안 맞는 경우가 생긴다). 못 찾으면 rows[0]으로 대체.
        match = next((row for row in rows if row[2] in answer), rows[0])
        c, d, f, fid, _ = match
        attachment = {
            "company": c,
            "doc_type": d,
            "file_name": f,
            "drive_url": f"https://drive.google.com/file/d/{fid}/view",
        }

    return {
        "answer": answer,
        "sources": [
            {
                "company": c,
                "doc_type": d,
                "file_name": f,
                # 소스 태그 클릭 시 실제 파일로 이동할 수 있게 Drive 링크도 함께 내려줌
                "drive_url": f"https://drive.google.com/file/d/{fid}/view",
            }
            for c, d, f, fid in sources
        ],
        "attachment": attachment,
    }


@app.get("/companies")
def companies():
    """사이드바 필터 목록을 하드코딩하지 않고 실제 DB에 자료가 있는 생명사만 내려준다."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT company FROM manual_chunks ORDER BY company")
        rows = cur.fetchall()
    conn.close()
    return {"companies": [r[0] for r in rows]}


# 실행: uvicorn rag_query:app --reload --port 8000
