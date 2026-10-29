from typing import List, Tuple
from ingestion.document_reader import DocumentReader
from ingestion.chunker import TextChunker
from retrieval.embedder import LocalEmbedder
from retrieval.vector_store import VectorStore
from generation.llm import LocalLLM
from ingestion.models import DocumentChunk

class RAGPipeline:
    """Orchestrates ingestion, retrieval, and generation."""
    
    def __init__(self, model_path: str):
        print("Initializing Embedder...")
        self.embedder = LocalEmbedder()
        
        print("Initializing FAISS Vector Store...")
        self.store = VectorStore(embedding_dim=self.embedder.dimension)
        
        print("Initializing Local LLM...")
        self.llm = LocalLLM(model_path=model_path)
        
        self.reader = DocumentReader()
        self.chunker = TextChunker(chunk_size=500, overlap=50)

    def ingest_document(self, file_path: str):
        """Extracts, chunks, embeds, and indexes a document."""
        print(f"\nIngesting {file_path}...")
        chunks = []
        for page in self.reader.read_pages(file_path):
            chunks.extend(list(self.chunker.chunk_page(page)))
            
        if not chunks:
            print("No text extracted.")
            return

        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        self.store.add(chunks, embeddings)
        print(f"Successfully indexed {len(chunks)} chunks.")

    def ask(self, question: str, k: int = 6) -> dict:
        """Retrieves context and generates an answer, returning both for diagnostics."""
        query_vector = self.embedder.embed_query(question)
        results = self.store.search(query_vector, k=k)
        
        if not results:
            return {"answer": "No relevant information found in the documents.", "context": []}

        # 1. Format the retrieved context for the LLM
        context_blocks = []
        context_data = [] # New: Store raw data for our API payload
        
        for rank, (chunk, distance) in enumerate(results):
            context_blocks.append(f"--- Excerpt {rank + 1} ---\n{chunk.text}")
            # Save the raw chunk text and metadata to send back to the UI
            context_data.append({
                "source": f"{chunk.source_file} (Page {chunk.page_num})",
                "distance": distance,
                "text": chunk.text
            })
            
        context_str = "\n\n".join(context_blocks)
        
        # 2. Construct the strict prompt
        system_prompt = (
            "You are a strict, precise answering assistant. "
            "Use ONLY the following context to answer the question. "
            "If the answer is not contained in the context, say 'I cannot answer this based on the provided documents.' "
            "Do not invent or hallucinate information.\n\n"
            f"CONTEXT:\n{context_str}"
        )
        
        # 3. Generate the answer
        answer = self.llm.generate_answer(system_prompt=system_prompt, user_prompt=question)
        
        # 4. Return both the LLM answer and the raw evidence
        return {
            "answer": answer,
            "context": context_data
        }