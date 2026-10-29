import os
import time
from ingestion.document_reader import DocumentReader
from ingestion.chunker import TextChunker
from retrieval.embedder import LocalEmbedder
from retrieval.vector_store import VectorStore

def main():
    sample_pdf = "Shivam Kumar.pdf"
    if not os.path.exists(sample_pdf):
        print("Please ensure sample.pdf is in the root directory.")
        return

    print("1. Loading Document...")
    reader = DocumentReader()
    chunker = TextChunker(chunk_size=500, overlap=50)
    
    chunks = []
    for page in reader.read_pages(sample_pdf):
        chunks.extend(list(chunker.chunk_page(page)))
    print(f"   Created {len(chunks)} chunks.")

    print("\n2. Initializing Embedding Model (First run may download weights)...")
    embedder = LocalEmbedder()
    store = VectorStore(embedding_dim=embedder.dimension)

    print("\n3. Embedding Chunks and building FAISS index...")
    t0 = time.time()
    # Extract just the string text for the embedder
    texts = [chunk.text for chunk in chunks]
    chunk_embeddings = embedder.embed_batch(texts)
    
    store.add(chunks, chunk_embeddings)
    t1 = time.time()
    print(f"   Indexed {store.index.ntotal} vectors in {t1-t0:.2f} seconds.")

    print("\n4. Performing Vector Search...")
    # EDIT THIS QUESTION to match something actually in your sample.pdf
    query = "What is the work experience mentioned in this document?"
    print(f"   Query: '{query}'")
    
    query_vector = embedder.embed_query(query)
    results = store.search(query_vector, k=3)

    print("\n--- TOP 3 RESULTS ---")
    for rank, (chunk, distance) in enumerate(results, 1):
        print(f"\nRank {rank} (L2 Distance: {distance:.4f})")
        print(f"Source: {chunk.source_file}, Page: {chunk.page_num}")
        print(f"Text snippet: {chunk.text[:200]}...")

if __name__ == "__main__":
    main()