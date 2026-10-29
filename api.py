from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import os
import tempfile
from pipeline import RAGPipeline

app = FastAPI(title="LocalMind API")

# Global pipeline instance
pipeline = None

class QueryRequest(BaseModel):
    question: str

@app.on_event("startup")
def load_model():
    global pipeline
    model_path = os.path.join(os.getcwd(), "models", "Llama-3.2-1B-Instruct-Q8_0.gguf")
    if not os.path.exists(model_path):
        raise RuntimeError("Model not found. Run download script.")
    pipeline = RAGPipeline(model_path=model_path)

@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Accepts a PDF file via HTTP and indexes it."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
            
        pipeline.ingest_document(tmp_path)
        os.remove(tmp_path)
        
        return {"status": "success", "message": f"Indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
def ask_question(query: QueryRequest):
    """Accepts a JSON question and returns the LLM answer."""
    try:
        answer = pipeline.ask(query.question)
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))