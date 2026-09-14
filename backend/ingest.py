"""
1단계: Google Drive 폴더를 재귀적으로 순회하며
       생명사별 파일 목록 + 메타데이터를 수집한다.

폴더 구조 가정 (통합 루트 폴더 하나, DRIVE_ROOT_FOLDER_ID):
  루트 폴더/
    청약 매뉴얼/
      삼성생명/파일1.pdf, 파일2.xlsx ...
      한화생명/...
    배서 매뉴얼/
      삼성생명/...
      ...
"""
import json
import os
import unicodedata
from dataclasses import dataclass, asdict
from typing import List
from google.oauth2 import service_account
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _normalize(name: str) -> str:
    """Drive(특히 macOS 동기화)가 한글 파일명을 NFD(자모 분리형)로 내려줄 때가 있어
    NFC(완성형)로 통일한다. .env 등에서 직접 입력하는 문자열은 보통 NFC이기 때문."""
    return unicodedata.normalize("NFC", name)


@dataclass
class DriveFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    company: str          # 생명사명 (하위 폴더명에서 추출)
    doc_type: str          # "청약" 또는 "배서"


def get_drive_service(service_account_json: str):
    """GOOGLE_SERVICE_ACCOUNT_JSON 값을 받아 Drive 서비스 객체를 만든다.
    로컬 개발 환경에서는 보통 service_account.json 파일 경로를 쓰지만,
    Railway 등 PaaS에서는 파일을 따로 올리기 번거로우므로 JSON 내용 자체를
    환경변수 값으로 넣어도 되도록 두 방식을 모두 지원한다."""
    stripped = service_account_json.strip()
    if stripped.startswith("{"):
        info = json.loads(stripped)
        creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    else:
        creds = service_account.Credentials.from_service_account_file(
            service_account_json, scopes=SCOPES
        )
    return build("drive", "v3", credentials=creds)


def list_subfolders(service, folder_id: str) -> dict:
    """폴더 바로 아래의 하위 폴더 목록 반환. {폴더명: folder_id}"""
    query = (
        f"'{folder_id}' in parents and "
        "mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    )
    results = service.files().list(q=query, fields="files(id, name)").execute()
    return {_normalize(f["name"]): f["id"] for f in results.get("files", [])}


# 하위 호환용 별칭
list_company_subfolders = list_subfolders


def resolve_doc_type_folders(service, root_folder_id: str, owner_tag: str | None = None) -> dict:
    """통합 루트 폴더(청약+배서) 아래에서 "생명사" 청약/배서 매뉴얼 폴더를 찾아
    {"청약": folder_id, "배서": folder_id} 형태로 반환한다.
    폴더명에 "생명사"와 "청약"/"배서"가 모두 포함되어야 매칭한다 (손보사 등 다른 폴더 제외).

    owner_tag가 주어지면 폴더명에 해당 문자열(예: "박준희")도 포함된 폴더만 후보로 삼는다.
    같은 문서 유형에 후보가 둘 이상이면(예: 담당자별 폴더가 여러 개) 에러를 발생시킨다."""
    subfolders = list_subfolders(service, root_folder_id)
    owner_tag = _normalize(owner_tag) if owner_tag else owner_tag
    candidates = {"청약": [], "배서": []}
    for name, folder_id in subfolders.items():
        if "생명사" not in name:
            continue
        if owner_tag and owner_tag not in name:
            continue
        if "청약" in name:
            candidates["청약"].append((name, folder_id))
        elif "배서" in name:
            candidates["배서"].append((name, folder_id))

    resolved = {}
    for doc_type, matches in candidates.items():
        if len(matches) > 1:
            names = [n for n, _ in matches]
            raise ValueError(
                f"'{doc_type}' 폴더 후보가 여러 개 발견되었습니다: {names}. "
                f"DRIVE_OWNER_TAG 환경변수로 대상을 좁혀주세요."
            )
        if matches:
            resolved[doc_type] = matches[0][1]

    missing = {"청약", "배서"} - resolved.keys()
    if missing:
        raise ValueError(
            f"루트 폴더({root_folder_id}) 아래에서 다음 하위 폴더를 찾지 못했습니다: {missing}. "
            f"실제 하위 폴더 목록: {list(subfolders.keys())}"
        )
    return resolved


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


def list_files_recursive(service, folder_id: str) -> List[dict]:
    """폴더와 그 아래 모든 하위 폴더를 재귀적으로 순회하며 파일 목록을 모은다.
    생명사 폴더 안에 연도/상품별 등으로 폴더가 더 나뉘어 있는 경우를 위함."""
    files = list_files_in_folder(service, folder_id)
    for sub_id in list_subfolders(service, folder_id).values():
        files.extend(list_files_recursive(service, sub_id))
    return files


def _skip_wrapper_folders(service, folder_id: str) -> str:
    """파일 없이 하위 폴더가 딱 하나뿐인 '래퍼' 폴더를 만나면 그 안으로 계속 내려간다.
    예: 생명사_청약매뉴얼(박준희)/생명사_청약매뉴얼/삼성생명/... 처럼
    문서유형 폴더와 생명사별 폴더 사이에 이름만 다른 폴더가 한 겹 더 있는 경우 대응."""
    while True:
        subs = list_subfolders(service, folder_id)
        files = list_files_in_folder(service, folder_id)
        if not files and len(subs) == 1:
            folder_id = next(iter(subs.values()))
            continue
        return folder_id


def collect_all_files(service, root_folder_id: str, doc_type: str) -> List[DriveFile]:
    """청약 또는 배서 매뉴얼 루트 폴더 전체를 순회하여 DriveFile 리스트로 반환"""
    root_folder_id = _skip_wrapper_folders(service, root_folder_id)
    collected = []
    companies = list_company_subfolders(service, root_folder_id)
    for company, folder_id in companies.items():
        for f in list_files_recursive(service, folder_id):
            collected.append(
                DriveFile(
                    file_id=f["id"],
                    name=_normalize(f["name"]),
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

    doc_type_folders = resolve_doc_type_folders(
        service, os.environ["DRIVE_ROOT_FOLDER_ID"], owner_tag=os.environ.get("DRIVE_OWNER_TAG")
    )

    sub_files = collect_all_files(service, doc_type_folders["청약"], "청약")
    endo_files = collect_all_files(service, doc_type_folders["배서"], "배서")

    all_files = sub_files + endo_files
    print(f"총 {len(all_files)}개 파일 발견")
    for f in all_files[:5]:
        print(asdict(f))
