import re
from typing import List

from app.ingestion.parsers.base import BaseParser, ParsedPage


class TextParser(BaseParser):
    def parse(self, raw: bytes, filename: str = "") -> List[ParsedPage]:
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception as e:
            raise RuntimeError(f"Text parsing failed for '{filename}': {e}") from e

        # Strip YAML frontmatter (for .md files)
        text = re.sub(r"^---.*?---\n", "", text, flags=re.DOTALL)

        if len(text.strip()) < 20:
            return []

        # Split markdown by headings for better section awareness
        sections = re.split(r"\n#{1,6} ", text)
        pages = []
        for i, section in enumerate(sections):
            section = section.strip()
            if len(section) > 20:
                pages.append(ParsedPage(text=section, page_num=None, section=None))

        return pages if pages else [ParsedPage(text=text.strip(), page_num=None)]
