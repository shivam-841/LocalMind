import os
from ingestion.document_reader import DocumentReader
from ingestion.chunker import TextChunker

def main():
    sample_pdf = "Shivam Kumar.pdf"
    
    if not os.path.exists(sample_pdf):
        print(f"Error: {sample_pdf} not found.")
        return

    reader = DocumentReader()
    chunker = TextChunker(chunk_size=500, overlap=50)
    
    print(f"Processing {sample_pdf}...\n")
    
    chunk_count = 0
    # Lazy evaluation: pages are processed one at a time
    for page in reader.read_pages(sample_pdf):
        for chunk in chunker.chunk_page(page):
            chunk_count += 1
            print(f"--- Chunk {chunk_count} (Page {chunk.page_num}) | Length: {chunk.length} chars ---")
            # Print a preview of the chunk
            preview = chunk.text.replace("\n", " ")[:100]
            print(f"{preview}...\n")
            
            if chunk_count >= 5:
                print("Stopping after 5 chunks for the preview.")
                return

if __name__ == "__main__":
    main()
