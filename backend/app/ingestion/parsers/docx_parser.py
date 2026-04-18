from io import BytesIO
from typing import List

from app.ingestion.parsers.base import BaseParser, ParsedPage


class DocxParser(BaseParser):
    def parse(self, raw: bytes, filename: str = "") -> List[ParsedPage]:
        try:
            from docx import Document
            doc = Document(BytesIO(raw))
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            text = "\n".join(paragraphs)
            if len(text) < 20:
                return []
            return [ParsedPage(text=text, page_num=None)]
        except Exception as e:
            raise RuntimeError(f"DOCX parsing failed for '{filename}': {e}") from e
