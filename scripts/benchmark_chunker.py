import time
import textwrap
from ingestion.chunker import TextChunker
from ingestion.models import DocumentChunk

def main():
    # Simulate a massive 10MB document (~3,000 pages of text)
    print("Generating 10MB dummy document...")
    dummy_text = "This is a test sentence designed to simulate actual document content. " * 150_000
    page = DocumentChunk(text=dummy_text, source_file="benchmark.pdf", page_num=1)
    
    # Test Python Fallback (temporarily force Python backend)
    chunker_py = TextChunker(chunk_size=1000, overlap=200)
    import ingestion.chunker
    ingestion.chunker.HAS_CPP_BACKEND = False
    
    print("Benchmarking Python String Slicing...")
    t0 = time.perf_counter()
    py_chunks = list(chunker_py.chunk_page(page))
    t_py = time.perf_counter() - t0
    
    # Test C++ Zero-Copy Backend
    ingestion.chunker.HAS_CPP_BACKEND = True
    chunker_cpp = TextChunker(chunk_size=1000, overlap=200)
    
    print("Benchmarking C++ Zero-Copy Boundaries...")
    t0 = time.perf_counter()
    cpp_chunks = list(chunker_cpp.chunk_page(page))
    t_cpp = time.perf_counter() - t0
    
    print("\n--- RESULTS ---")
    print(f"Chunks generated: {len(cpp_chunks)}")
    print(f"Python time: {t_py:.4f} seconds")
    print(f"C++ time:    {t_cpp:.4f} seconds")
    if t_cpp < t_py:
        print(f"Speedup:     {t_py / t_cpp:.2f}x faster")

if __name__ == "__main__":
    main()