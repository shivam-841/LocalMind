import pymupdf
from typing import Iterator
from ingestion.models import DocumentChunk

class DocumentReader:
    """Handles text and metadata extraction from local documents."""
    
    def read_pages(self, file_path: str) -> Iterator[DocumentChunk]:
        """
        Extracts raw text from a PDF file.
        
        Args:
            file_path (str): The absolute or relative path to the PDF.
            
        Yields:
            DocumentChunk: The text and metadata for each non-empty page.
        """
        extracted_text = []
        
        # The 'with' block acts like RAII for the file handle
        with pymupdf.open(file_path) as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text().strip()
                
                if text: # Skip completely empty pages
                    # PyMuPDF is 0-indexed, but humans read 1-indexed pages
                    yield DocumentChunk(
                        text=text,
                        source_file=file_path,
                        page_num=page_num + 1 
                    )

    def extract_text(self, file_path: str) -> str:
        """Extract and concatenate text from every non-empty page in a PDF."""
        return "\n".join(page.text for page in self.read_pages(file_path))
