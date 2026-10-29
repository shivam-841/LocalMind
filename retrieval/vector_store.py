import faiss
import numpy as np
from typing import List, Tuple
from ingestion.models import DocumentChunk

class VectorStore:
    """In-memory C++ powered vector index using FAISS."""
    
    def __init__(self, embedding_dim: int):
        # IndexFlatL2 performs an exact brute-force L2 (Euclidean) distance search.
        # Since we are working with < 100,000 chunks, brute-force in C++ is instant.
        self.index = faiss.IndexFlatL2(embedding_dim)
        # We keep the Python objects in a list. The list index matches the FAISS vector ID.
        self.chunks: List[DocumentChunk] = []

    def add(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Adds chunks and their embeddings to the index."""
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("Number of chunks must match number of embeddings.")
            
        # Ensure memory layout is C-contiguous for the C++ backend
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[DocumentChunk, float]]:
        """Finds the top-k most similar chunks to the query."""
        if self.index.ntotal == 0:
            return []
            
        # FAISS expects a 2D array (matrix) for batch processing, even for 1 query.
        query_embedding = np.ascontiguousarray(query_embedding.reshape(1, -1), dtype=np.float32)
        
        # distances are L2 distances (lower is better/closer)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append((self.chunks[idx], float(distances[0][i])))
                
        return results