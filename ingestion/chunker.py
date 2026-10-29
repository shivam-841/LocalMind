from typing import Iterator
from ingestion.models import DocumentChunk

try:
    import localmind_cpp
    HAS_CPP_BACKEND = True
except ImportError:
    HAS_CPP_BACKEND = False

class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_page(self, page: DocumentChunk) -> Iterator[DocumentChunk]:
        text = page.text
        
        if HAS_CPP_BACKEND:
            # C++ returns lightweight integer tuples: [(start, length), ...]
            boundaries = localmind_cpp.get_chunk_boundaries(text, self.chunk_size, self.overlap)
            
            for start, length in boundaries:
                # Python allocates the final chunk string exactly once here
                chunk_text = text[start : start + length]
                yield DocumentChunk(
                    text=chunk_text,
                    source_file=page.source_file,
                    page_num=page.page_num
                )
        else:
            # Fallback Python logic
            start = 0
            text_length = len(text)

            while start < text_length:
                end = start + self.chunk_size
                chunk_text = text[start:end].strip()
                
                if chunk_text:
                    yield DocumentChunk(
                        text=chunk_text,
                        source_file=page.source_file,
                        page_num=page.page_num
                    )
                
                start += (self.chunk_size - self.overlap)