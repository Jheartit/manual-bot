"""
1단계: Google Drive 폴더를 재귀적으로 순회하며
       생명사별 파일 목록 + 메타데이터를 수집한다.

폴더 구조 가정:
  생명사 청약 매뉴얼/
    삼성생명/파일1.pdf, 파일2.xlsx ...
    한화생명/...
  생명사 배서 매뉴얼/
    삼성생명/...
    ...
"""
import os
from dataclasses import dataclass, asdict
from typing import List
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    company: str          # 생명사명 (하위 폴더명에서 추출)
    doc_type: str          # "청약" 또는 "배서"


def get_drive_service(service_account_json: str):
    creds = service_account.Credentials.from_service_account_file(
        service_account_json, scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def list_company_subfolders(service, root_folder_id: str) -> dict:
    """루트 폴더(청약/배서 매뉴얼) 아래의 생명사별 하위 폴더를 반환. {생명사명: folder_id}"""
    query = (
        f"'{root_folder_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {f["name"]: f["id"] for f in results.get("files", [])}


def list_files_in_folder(service, folder_id: str) -> List[dict]:
    """폴더 내 파일 목록 (폴더 제외)"""
    files = []
    page_token = None
    query = f"'{folder_id}' in parents and mimeType != 'application/vnd.google-apps.folder' and trashed = false"
    while True:
        resp = service.files().list(
            q=query,
            fields="nextPageToken, files(id, name, mimeType, modifiedTime)",
            pageToken=page_token,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return files


def collect_all_files(service, root_folder_id: str, doc_type: str) -> List[DriveFile]:
    """청약 또는 배서 매뉴얼 루트 폴더 전체를 순회하여 DriveFile 리스트로 반환"""
    collected = []
    companies = list_company_subfolders(service, root_folder_id)
    for company, folder_id in companies.items():
        for f in list_files_in_folder(service, folder_id):
            collected.append(
                DriveFile(
                    file_id=f["id"],
                    name=f["name"],
                    mime_type=f["mimeType"],
                    modified_time=f["modifiedTime"],
                    company=company,
                    doc_type=doc_type,
                )
            )
    return collected


def download_file(service, file_id: str, dest_path: str):
    """파일 바이너리 다운로드 (PDF/이미지/xlsx 모두 동일하게 처리)"""
    request = service.files().get_media(fileId=file_id)
    with open(dest_path, "wb") as fh:
        from googleapiclient.http import MediaIoBaseDownload
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    service = get_drive_service(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])

    sub_files = collect_all_files(
        service, os.environ["DRIVE_FOLDER_ID_SUBSCRIPTION"], "청약"
    )
    endo_files = collect_all_files(
        service, os.environ["DRIVE_FOLDER_ID_ENDORSEMENT"], "배서"
    )

    all_files = sub_files + endo_files
    print(f"총 {len(all_files)}개 파일 발견")
    for f in all_files[:5]:
        print(asdict(f))
