"""
3단계: 파싱된 텍스트를 청크로 나누고 임베딩하여 pgvector에 저장한다.
"""
import os
import psycopg2
import voyageai
from typing import List
from dotenv import load_dotenv

load_dotenv()  # 이 모듈이 어디서 import되든 클라이언트 생성 전에 .env가 로드되도록 보장
vo = voyageai.Client()  # VOYAGE_API_KEY 환경변수 사용

CHUNK_SIZE = 800       # 대략적인 글자 수 기준 (토큰 아님, 러프한 근사치)
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """단순 슬라이딩 윈도우 청킹. 실제 운영시엔 섹션/조항 경계를 고려한 청킹 권장."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


EMBED_BATCH_SIZE = 128  # 한 번에 너무 많은 청크를 보내면 요청 제한(TPM 등)에 걸릴 수 있어 나눠 보낸다


def embed_chunks(chunks: List[str]) -> List[list]:
    embeddings = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        result = vo.embed(batch, model="voyage-3", input_type="document")
        embeddings.extend(result.embeddings)
    return embeddings


def init_db(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manual_chunks (
                id SERIAL PRIMARY KEY,
                company TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                drive_file_id TEXT NOT NULL,
                chunk_index INT NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(1024)
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS manual_chunks_embedding_idx
            ON manual_chunks USING ivfflat (embedding vector_cosine_ops);
        """)
        # 증분 처리를 위해 파일별 마지막 처리 시점(modified_time)을 기록
        cur.execute("""
            CREATE TABLE IF NOT EXISTS manual_files (
                drive_file_id TEXT PRIMARY KEY,
                company TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                file_name TEXT NOT NULL,
                modified_time TEXT NOT NULL
            );
        """)
    conn.commit()


def get_processed_modified_times(conn) -> dict:
    """이미 처리된 파일들의 {drive_file_id: modified_time} 반환. 증분 처리 시 변경 여부 판단에 사용."""
    with conn.cursor() as cur:
        cur.execute("SELECT drive_file_id, modified_time FROM manual_files")
        return dict(cur.fetchall())


def delete_file(conn, drive_file_id: str):
    """Drive에서 사라진 파일의 청크/기록을 정리"""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM manual_chunks WHERE drive_file_id = %s", (drive_file_id,))
        cur.execute("DELETE FROM manual_files WHERE drive_file_id = %s", (drive_file_id,))
    conn.commit()


def store_document(conn, company: str, doc_type: str, file_name: str,
                    drive_file_id: str, full_text: str, modified_time: str):
    """파싱된 문서 전체 텍스트를 청킹→임베딩→DB저장까지 한번에 처리"""
    chunks = chunk_text(full_text)
    if not chunks:
        return
    embeddings = embed_chunks(chunks)

    with conn.cursor() as cur:
        # 재처리 시 기존 청크 삭제 (같은 파일 재수집 대비)
        cur.execute("DELETE FROM manual_chunks WHERE drive_file_id = %s", (drive_file_id,))
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """INSERT INTO manual_chunks
                   (company, doc_type, file_name, drive_file_id, chunk_index, content, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (company, doc_type, file_name, drive_file_id, i, chunk, emb),
            )
        cur.execute(
            """INSERT INTO manual_files (drive_file_id, company, doc_type, file_name, modified_time)
               VALUES (%s, %s, %s, %s, %s)
               ON CONFLICT (drive_file_id) DO UPDATE SET
                   company = EXCLUDED.company,
                   doc_type = EXCLUDED.doc_type,
                   file_name = EXCLUDED.file_name,
                   modified_time = EXCLUDED.modified_time""",
            (drive_file_id, company, doc_type, file_name, modified_time),
        )
    conn.commit()
    print(f"  저장 완료: {file_name} ({len(chunks)} 청크)")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    init_db(conn)
    print("DB 초기화 완료 (manual_chunks 테이블 생성)")
