import os
import time
from generation.llm import LocalLLM

def main():
    model_path = os.path.join(os.getcwd(), "models", "Llama-3.2-1B-Instruct-Q8_0.gguf")
    
    if not os.path.exists(model_path):
        print("Model not found. Run 'python scripts/download_model.py' first.")
        return

    print("1. Loading Llama-3.2 into RAM (via llama.cpp)...")
    t0 = time.time()
    llm = LocalLLM(model_path=model_path)
    print(f"   Loaded in {time.time() - t0:.2f} seconds.")

    print("\n2. Generating response...")
    system_prompt = "You are a helpful, precise engineering assistant."
    user_prompt = "Explain in one short sentence what a C++ virtual destructor does."
    
    t1 = time.time()
    answer = llm.generate_answer(system_prompt, user_prompt)
    t2 = time.time()
    
    print("\n--- LLM RESPONSE ---")
    print(answer)
    print("--------------------")
    print(f"Generation took {t2 - t1:.2f} seconds.")

if __name__ == "__main__":
    main()