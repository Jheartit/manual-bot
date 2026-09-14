"""
전체 파이프라인 실행: Drive 수집 → 다운로드 → 파싱 → 청킹/임베딩 → DB 저장

매일 1회 cron으로 실행하면 됨. Drive의 modified_time을 DB(manual_files)에 저장된
값과 비교해 변경된 파일만 재처리하고, Drive에서 사라진 파일은 정리한다.
전체를 강제로 다시 처리하려면 --full 옵션을 준다.
"""
import os
import sys
import tempfile
import psycopg2
from dotenv import load_dotenv

# parse/embed_store는 import 시점에 OpenAI/Voyage 클라이언트를 생성하므로
# 반드시 그 전에 .env를 읽어둬야 한다.
load_dotenv()

from ingest import get_drive_service, collect_all_files, download_file, resolve_doc_type_folders
from parse import parse_file
from embed_store import init_db, store_document, get_processed_modified_times, delete_file


def _connect():
    return psycopg2.connect(os.environ["DATABASE_URL"])


def main():
    force_full = "--full" in sys.argv

    service = get_drive_service(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    conn = _connect()
    init_db(conn)

    doc_type_folders = resolve_doc_type_folders(
        service, os.environ["DRIVE_ROOT_FOLDER_ID"], owner_tag=os.environ.get("DRIVE_OWNER_TAG")
    )
    sub_files = collect_all_files(service, doc_type_folders["청약"], "청약")
    endo_files = collect_all_files(service, doc_type_folders["배서"], "배서")
    all_files = sub_files + endo_files

    processed_times = {} if force_full else get_processed_modified_times(conn)

    # 변경/신규 파일만 골라서 처리 대상으로 삼음
    to_process = [
        f for f in all_files
        if processed_times.get(f.file_id) != f.modified_time
    ]
    skipped = len(all_files) - len(to_process)

    print(f"총 {len(all_files)}개 파일 중 {len(to_process)}개 신규/변경, {skipped}개 변경 없음(건너뜀)")

    for f in to_process:
        print(f"[{f.company}/{f.doc_type}] {f.name} 처리 중...")
        try:
            with tempfile.NamedTemporaryFile(suffix="_" + f.name, delete=False) as tmp:
                download_file(service, f.file_id, tmp.name)
                text = parse_file(tmp.name, f.mime_type)

            try:
                store_document(
                    conn,
                    company=f.company,
                    doc_type=f.doc_type,
                    file_name=f.name,
                    drive_file_id=f.file_id,
                    full_text=text,
                    modified_time=f.modified_time,
                )
            except (psycopg2.OperationalError, psycopg2.InterfaceError):
                # 오래 걸리는 파이프라인 도중 idle timeout 등으로 DB 연결이 끊길 수 있다.
                # 한 번 재연결해서 그 파일만 재시도하고, 그래도 실패하면 다음 파일로 넘어간다.
                print("  DB 연결이 끊겨 재연결 후 재시도합니다...")
                conn = _connect()
                store_document(
                    conn,
                    company=f.company,
                    doc_type=f.doc_type,
                    file_name=f.name,
                    drive_file_id=f.file_id,
                    full_text=text,
                    modified_time=f.modified_time,
                )
        except Exception as e:
            print(f"  실패: {f.name} - {e}")

    # Drive에서 삭제/이동된 파일은 DB에서도 정리
    current_ids = {f.file_id for f in all_files}
    removed_ids = set(processed_times.keys()) - current_ids
    for file_id in removed_ids:
        try:
            delete_file(conn, file_id)
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            conn = _connect()
            delete_file(conn, file_id)
    if removed_ids:
        print(f"Drive에서 사라진 {len(removed_ids)}개 파일의 데이터 삭제 완료")

    conn.close()
    print("파이프라인 완료")


if __name__ == "__main__":
    main()
