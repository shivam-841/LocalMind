import os
from pipeline import RAGPipeline

def main():
    model_path = os.path.join(os.getcwd(), "models", "Llama-3.2-1B-Instruct-Q8_0.gguf")
    sample_pdf = "Shivam Kumar.pdf"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    if not os.path.exists(sample_pdf):
        print(f"Error: {sample_pdf} not found.")
        return

    print("--- Starting LocalMind ---")
    pipeline = RAGPipeline(model_path=model_path)
    
    pipeline.ingest_document(sample_pdf)
    
    print("\n--- LocalMind Ready ---")
    print("Type your questions below (or 'quit' to exit).")
    
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ['quit', 'exit', 'q']:
            break
        if not question:
            continue
            
        print("\nThinking...")
        answer = pipeline.ask(question)
        
        print("\n" + "="*40)
        print(answer)
        print("="*40)

if __name__ == "__main__":
    main()