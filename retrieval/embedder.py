from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List

class LocalEmbedder:
    """Wrapper for local CPU-friendly embedding models."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # On first run, this downloads the model weights to ~/.cache/huggingface
        # Subsequent runs load it directly from disk into RAM.
        self.model = SentenceTransformer(model_name)
        
    @property
    def dimension(self) -> int:
        """Returns the size of the vector (e.g., 384)."""
        return self.model.get_embedding_dimension()
        
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Converts a list of strings into a 2D numpy array of vectors."""
        # encode() returns a numpy array. We force float32 for FAISS compatibility.
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embeds a single search query."""
        embedding = self.model.encode(query, convert_to_numpy=True)
        return embedding.astype(np.float32)