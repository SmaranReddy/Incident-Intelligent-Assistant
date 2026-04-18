from app.ingestion.parsers.base import ParsedPage, BaseParser
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.text_parser import TextParser
from app.ingestion.parsers.docx_parser import DocxParser


SUPPORTED_TYPES = {"pdf", "txt", "md", "docx"}


def get_parser(file_type: str) -> BaseParser:
    parsers = {
        "pdf": PDFParser,
        "txt": TextParser,
        "md": TextParser,
        "docx": DocxParser,
    }
    cls = parsers.get(file_type.lower())
    if cls is None:
        raise ValueError(f"Unsupported file type: {file_type}")
    return cls()
