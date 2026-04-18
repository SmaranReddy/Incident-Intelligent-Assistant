from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ParsedPage:
    text: str
    page_num: Optional[int] = None
    section: Optional[str] = None


class BaseParser(ABC):
    @abstractmethod
    def parse(self, raw: bytes, filename: str = "") -> List[ParsedPage]:
        """Parse raw file bytes into a list of ParsedPage objects."""
        ...
