from llama_cpp import Llama

class LocalLLM:
    """Wrapper for local C++ inference using llama.cpp"""
    
    def __init__(self, model_path: str, n_ctx: int = 2048):
        """
        Args:
            model_path: Absolute path to the .gguf model file.
            n_ctx: The context window size (how many tokens it can remember at once).
        """
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=4, # Adjust based on your CPU cores
            verbose=False # Set to True if you want to see C++ memory allocation logs
        )

    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Sends a prompt to the model and returns the text response."""
        
        # Llama-3 formatting structure
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"        
        response = self.llm(
            prompt,
            max_tokens=256,
            temperature=0.1, # Low temperature = factual, less creative hallucination
            stop=["<|eot_id|>"],
            echo=False
        )
        
        return response["choices"][0]["text"].strip()