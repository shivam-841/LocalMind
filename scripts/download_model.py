from huggingface_hub import hf_hub_download
import os

def main():
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    print("Downloading Llama-3.2-1B-Instruct (Q8_0 GGUF)...")
    print("This is ~1.3 GB and may take a few minutes.")
    
    # We download a quantized GGUF model optimized for CPU execution
    model_path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
        filename="Llama-3.2-1B-Instruct-Q8_0.gguf",
        local_dir=model_dir
    )
    
    print(f"\nDownload complete! Model saved to:\n{model_path}")

if __name__ == "__main__":
    main()