from dataclasses import dataclass

@dataclass
class DocumentChunk:
    """Represents a piece of text and its source metadata."""
    text: str
    source_file: str
    page_num: int
    
    # We will use this later to measure how many characters we are processing
    @property
    def length(self) -> int:
        return len(self.text)