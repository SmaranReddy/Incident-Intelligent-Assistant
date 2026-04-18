from typing import List

from app.ingestion.parsers.base import BaseParser, ParsedPage


class PDFParser(BaseParser):
    def parse(self, raw: bytes, filename: str = "") -> List[ParsedPage]:
        try:
            import pymupdf  # PyMuPDF
            doc = pymupdf.open(stream=raw, filetype="pdf")
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text().strip()
                if len(text) > 20:
                    pages.append(ParsedPage(text=text, page_num=i + 1))
            doc.close()
            return pages
        except Exception as e:
            raise RuntimeError(f"PDF parsing failed for '{filename}': {e}") from e
