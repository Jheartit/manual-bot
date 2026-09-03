"""
2단계: 파일 형식별로 텍스트를 추출한다.
  - PDF (텍스트 기반): pdfplumber
  - PDF (스캔본, 텍스트 추출 실패 시): 페이지를 이미지로 렌더링 후 Claude Vision으로 OCR
  - 이미지: Claude Vision으로 OCR
  - xlsx: openpyxl로 시트별 파싱 → 마크다운 표로 변환
"""
import base64
import io
import fitz  # PyMuPDF
import pdfplumber
import openpyxl
from anthropic import Anthropic

client = Anthropic()  # ANTHROPIC_API_KEY 환경변수 사용


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
    """스캔본 PDF를 페이지 이미지로 변환 후 Claude Vision으로 텍스트 추출"""
    doc = fitz.open(path)
    results = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": "image/png", "data": img_b64
                    }},
                    {"type": "text", "text": (
                        "이 이미지는 보험 청약/배서 매뉴얼의 한 페이지입니다. "
                        "표는 마크다운 표 형식으로, 나머지는 원문 그대로 텍스트로만 추출해줘. "
                        "설명이나 요약은 하지 말고 원문 내용만 출력해."
                    )}
                ]
            }]
        )
        page_text = "".join(
            block.text for block in resp.content if block.type == "text"
        )
        results.append(f"[페이지 {i+1}]\n{page_text}")
    return "\n\n".join(results)


def parse_image(path: str) -> str:
    """단일 이미지 파일 OCR (Claude Vision)"""
    with open(path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    ext = path.lower().split(".")[-1]
    media_type = "image/png" if ext == "png" else "image/jpeg"

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64", "media_type": media_type, "data": img_b64
                }},
                {"type": "text", "text": (
                    "이 이미지의 텍스트/표 내용을 그대로 추출해줘. "
                    "표는 마크다운 표 형식으로 정리해줘."
                )}
            ]
        }]
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def parse_xlsx(path: str) -> str:
    """xlsx의 모든 시트를 마크다운 표로 변환"""
    wb = openpyxl.load_workbook(path, data_only=True)
    output = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))
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


def parse_file(path: str, mime_type: str) -> str:
    """파일 형식에 따라 적절한 파서로 라우팅"""
    if mime_type == "application/pdf":
        return parse_pdf(path)
    elif mime_type.startswith("image/"):
        return parse_image(path)
    elif "spreadsheet" in mime_type or path.endswith(".xlsx"):
        return parse_xlsx(path)
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {mime_type}")
