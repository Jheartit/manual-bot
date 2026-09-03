"""
전체 파이프라인 실행: Drive 수집 → 다운로드 → 파싱 → 청킹/임베딩 → DB 저장

매일 1회 cron으로 실행하면 됨. modified_time을 DB와 비교해
변경된 파일만 재처리하도록 확장 가능 (여기서는 단순화를 위해 전체 재처리 버전으로 작성).
"""
import os
import tempfile
import psycopg2
from dotenv import load_dotenv

from ingest import get_drive_service, collect_all_files, download_file
from parse import parse_file
from embed_store import init_db, store_document

load_dotenv()


def main():
    service = get_drive_service(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    init_db(conn)

    sub_files = collect_all_files(service, os.environ["DRIVE_FOLDER_ID_SUBSCRIPTION"], "청약")
    endo_files = collect_all_files(service, os.environ["DRIVE_FOLDER_ID_ENDORSEMENT"], "배서")
    all_files = sub_files + endo_files

    print(f"총 {len(all_files)}개 파일 처리 시작")

    for f in all_files:
        print(f"[{f.company}/{f.doc_type}] {f.name} 처리 중...")
        try:
            with tempfile.NamedTemporaryFile(suffix="_" + f.name, delete=False) as tmp:
                download_file(service, f.file_id, tmp.name)
                text = parse_file(tmp.name, f.mime_type)

            store_document(
                conn,
                company=f.company,
                doc_type=f.doc_type,
                file_name=f.name,
                drive_file_id=f.file_id,
                full_text=text,
            )
        except Exception as e:
            print(f"  실패: {f.name} - {e}")

    conn.close()
    print("파이프라인 완료")


if __name__ == "__main__":
    main()
