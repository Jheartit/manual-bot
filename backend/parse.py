"""
2단계: 파일 형식별로 텍스트를 추출한다.
  - PDF (텍스트 기반): pdfplumber
  - PDF (스캔본, 텍스트 추출 실패 시): 페이지를 이미지로 렌더링 후 GPT-4o Vision으로 OCR
  - 이미지: GPT-4o Vision으로 OCR (tif/bmp 등은 Pillow로 png 변환 후 전달)
  - xlsx: openpyxl로 시트별 파싱 → 마크다운 표로 변환
  - pptx: python-pptx로 슬라이드별 텍스트/표 추출
  - docx: python-docx로 문단/표 추출
  - doc (구 버전 워드): macOS textutil로 변환 (다른 환경에는 도구가 없어 실패할 수 있음)
  - zip: 압축 해제 후 내부 파일들을 재귀적으로 파싱해 합침
  - hwp: OLE 구조를 직접 읽어 본문 텍스트만 추출하는 최소 파서 (표/서식 등은 무시,
    간단한 양식 문서 기준으로 검증됨. 복잡한 문서는 일부 텍스트 유실 가능)
"""
import base64
import io
import mimetypes
import shutil
import struct
import subprocess
import tempfile
import zipfile
import zlib
import fitz  # PyMuPDF
import olefile
import pdfplumber
import openpyxl
from docx import Document
from pptx import Presentation
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 이 모듈이 어디서 import되든 클라이언트 생성 전에 .env가 로드되도록 보장
# timeout을 지정하지 않으면 네트워크 문제 등으로 요청이 응답 없이 무한정 걸릴 수 있어
# 파이프라인 전체가 멈춰버린다. 페이지 1장 OCR에 90초면 충분하므로 짧게 제한한다.
client = OpenAI(timeout=90.0, max_retries=2)  # OPENAI_API_KEY 환경변수 사용
VISION_MODEL = "gpt-4o"


def parse_pdf(path: str) -> str:
    """텍스트 기반 PDF 우선 시도, 텍스트가 거의 없으면 스캔본으로 간주해 OCR로 전환"""
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            text_parts.append(t)

    total_text = "\n".join(text_parts).strip()

    # 추출된 텍스트가 페이지 수 대비 너무 적으면 스캔본으로 판단
    if len(total_text) < 50 * len(text_parts):
        return ocr_pdf_via_vision(path)

    return total_text


def ocr_pdf_via_vision(path: str, max_pages: int = 30) -> str:
    """스캔본 PDF를 페이지 이미지로 변환 후 GPT-4o Vision으로 텍스트 추출"""
    doc = fitz.open(path)
    results = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        resp = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "이 이미지는 보험 청약/배서 매뉴얼의 한 페이지입니다. "
                        "표는 마크다운 표 형식으로, 나머지는 원문 그대로 텍스트로만 추출해줘. "
                        "설명이나 요약은 하지 말고 원문 내용만 출력해."
                    )},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_b64}"
                    }},
                ]
            }]
        )
        page_text = resp.choices[0].message.content or ""
        results.append(f"[페이지 {i+1}]\n{page_text}")
    return "\n\n".join(results)


def parse_image(path: str) -> str:
    """단일 이미지 파일 OCR (GPT-4o Vision).
    GPT Vision은 png/jpeg/gif/webp만 지원하므로, tif/bmp 등은 Pillow로 png로 변환한다."""
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    resp = client.chat.completions.create(
        model=VISION_MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "이 이미지의 텍스트/표 내용을 그대로 추출해줘. "
                    "표는 마크다운 표 형식으로 정리해줘."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:image/png;base64,{img_b64}"
                }},
            ]
        }]
    )
    return resp.choices[0].message.content or ""


def _is_blank_row(row) -> bool:
    return all(c is None or str(c).strip() == "" for c in row)


def parse_xlsx(path: str) -> str:
    """xlsx의 모든 시트를 마크다운 표로 변환.
    엑셀은 실제 데이터가 없어도 '사용된 범위'가 수십만 행으로 잡히는 경우가 흔해서
    (서식만 적용된 빈 셀 등), 완전히 빈 행은 걸러내지 않으면 텍스트가 비정상적으로
    커져 임베딩 API 요청 제한에 걸릴 수 있다."""
    wb = openpyxl.load_workbook(path, data_only=True)
    output = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = [r for r in ws.iter_rows(values_only=True) if not _is_blank_row(r)]
        if not rows:
            continue
        output.append(f"### 시트: {sheet_name}\n")
        header = rows[0]
        output.append("| " + " | ".join(str(c) if c is not None else "" for c in header) + " |")
        output.append("|" + "---|" * len(header))
        for row in rows[1:]:
            output.append("| " + " | ".join(str(c) if c is not None else "" for c in row) + " |")
        output.append("")
    return "\n".join(output)


def parse_pptx(path: str) -> str:
    """pptx의 슬라이드별 텍스트/표를 추출"""
    prs = Presentation(path)
    slides_out = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    text = "".join(run.text for run in para.runs)
                    if text.strip():
                        lines.append(text)
            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    lines.append(" | ".join(cell.text for cell in row.cells))
        if lines:
            slides_out.append(f"[슬라이드 {i}]\n" + "\n".join(lines))
    return "\n\n".join(slides_out)


def parse_docx(path: str) -> str:
    """docx의 문단과 표를 순서대로 텍스트로 추출"""
    doc = Document(path)
    parts = []
    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text)
    for table in doc.tables:
        rows = [[cell.text for cell in row.cells] for row in table.rows]
        if not rows:
            continue
        parts.append("| " + " | ".join(rows[0]) + " |")
        parts.append("|" + "---|" * len(rows[0]))
        for row in rows[1:]:
            parts.append("| " + " | ".join(row) + " |")
    return "\n".join(parts)


def parse_doc(path: str) -> str:
    """구버전 바이너리 .doc. macOS의 textutil로 텍스트 변환 (다른 OS에는 도구가 없어 실패할 수 있음)"""
    if not shutil.which("textutil"):
        raise ValueError(
            ".doc 파일은 이 환경에 변환 도구(textutil)가 없어 처리할 수 없습니다. "
            "워드에서 .docx로 '다른 이름으로 저장' 후 다시 업로드해주세요."
        )
    with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
        subprocess.run(
            ["textutil", "-convert", "txt", "-output", tmp.name, path],
            check=True, capture_output=True,
        )
        with open(tmp.name, "r", encoding="utf-8") as f:
            return f.read()


HWPTAG_PARA_TEXT = 67  # HWPTAG_BEGIN(0x10) + 51


def parse_hwp(path: str) -> str:
    """HWP 5.0 바이너리 포맷에서 본문 텍스트만 추출하는 최소 파서.
    OLE 복합 문서의 BodyText/SectionN 스트림을 열어 PARA_TEXT 레코드만 읽는다.
    표/이미지/글머리 등 서식 정보는 반영하지 않으므로 원본 레이아웃과는 다르다."""
    ole = olefile.OleFileIO(path)
    try:
        header = ole.openstream("FileHeader").read()
        flags = struct.unpack_from("<I", header, 36)[0]
        compressed = bool(flags & 1)

        section_streams = [
            entry for entry in ole.listdir()
            if len(entry) == 2 and entry[0] == "BodyText" and entry[1].startswith("Section")
        ]
        section_streams.sort(key=lambda e: int(e[1].replace("Section", "")))

        texts = []
        for entry in section_streams:
            data = ole.openstream(entry).read()
            if compressed:
                data = zlib.decompressobj(-15).decompress(data)
            texts.append(_extract_hwp_section_text(data))
        return "\n\n".join(t for t in texts if t.strip())
    finally:
        ole.close()


def _extract_hwp_section_text(data: bytes) -> str:
    out = []
    pos = 0
    n = len(data)
    while pos + 4 <= n:
        header = struct.unpack_from("<I", data, pos)[0]
        tag_id = header & 0x3FF
        size = (header >> 20) & 0xFFF
        pos += 4
        if size == 0xFFF:
            if pos + 4 > n:
                break
            size = struct.unpack_from("<I", data, pos)[0]
            pos += 4
        record = data[pos:pos + size]
        pos += size
        if tag_id == HWPTAG_PARA_TEXT:
            out.append(_decode_hwp_para_text(record))
    return "\n".join(t for t in out if t.strip())


def _decode_hwp_para_text(record: bytes) -> str:
    """PARA_TEXT 레코드는 UTF-16LE. 32 미만 코드는 탭/개행을 빼면 인라인 컨트롤
    문자(표·그림 등)이며 보통 뒤에 7워드(14바이트)의 부가데이터가 따라온다."""
    chars = []
    i = 0
    n = len(record)
    while i + 2 <= n:
        code = struct.unpack_from("<H", record, i)[0]
        i += 2
        if code == 0:
            continue
        if code < 32:
            if code == 9:
                chars.append("\t")
            elif code in (10, 13):
                chars.append("\n")
            else:
                i += 14
            continue
        chars.append(chr(code))
    return "".join(chars).strip()


def parse_zip(path: str) -> str:
    """zip 압축을 풀어 내부 파일들을 형식별로 재귀 파싱 후 하나로 합침"""
    parts = []
    with zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp_dir:
        for info in zf.infolist():
            if info.is_dir():
                continue
            inner_mime, _ = mimetypes.guess_type(info.filename)
            if not inner_mime:
                continue
            try:
                extracted_path = zf.extract(info, tmp_dir)
                text = parse_file(extracted_path, inner_mime)
                if text.strip():
                    parts.append(f"[{info.filename}]\n{text}")
            except Exception as e:
                parts.append(f"[{info.filename}] 처리 실패: {e}")
    return "\n\n---\n\n".join(parts)


def parse_file(path: str, mime_type: str) -> str:
    """파일 형식에 따라 적절한 파서로 라우팅"""
    if mime_type == "application/pdf":
        return parse_pdf(path)
    elif mime_type.startswith("image/"):
        return parse_image(path)
    elif "spreadsheet" in mime_type or path.endswith(".xlsx"):
        return parse_xlsx(path)
    elif "presentationml" in mime_type or path.endswith(".pptx"):
        return parse_pptx(path)
    elif "wordprocessingml" in mime_type or path.endswith(".docx"):
        return parse_docx(path)
    elif mime_type == "application/msword" or path.endswith(".doc"):
        return parse_doc(path)
    elif mime_type == "application/zip" or path.endswith(".zip"):
        return parse_zip(path)
    elif "hwp" in mime_type or path.endswith(".hwp"):
        return parse_hwp(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {mime_type}")
