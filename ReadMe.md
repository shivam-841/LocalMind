# 🧠 LocalMind
**A Privacy-First, Zero-Copy Local RAG & Document Intelligence System**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![C++](https://img.shields.io/badge/C++-20-00599C.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg)
![llama.cpp](https://img.shields.io/badge/Inference-llama.cpp-black.svg)

LocalMind is a fully offline Retrieval-Augmented Generation (RAG) pipeline engineered to extract, index, and query PDF documents using local language models. It is designed to operate strictly within hardware-constrained environments (8GB - 16GB RAM, CPU-only) without relying on external cloud APIs.

![LocalMind UI](assets/hero.png)

## 🏗️ System Architecture

LocalMind utilizes a decoupled microservice architecture, separating the heavy LLM inference backend from the client UI, while bridging C++ and Python for memory-efficient document ingestion.

```mermaid
graph TD
    subgraph Frontend [Client UI]
        A[Streamlit Web App]
    end

    subgraph Backend [FastAPI Microservice]
        B[PDF Upload Endpoint]
        C[Query Endpoint]
        
        subgraph C++ Engine [Ingestion]
            D[pybind11 Boundary]
            E[std::string_view Zero-Copy Chunker]
        end
        
        subgraph Vector Engine [Retrieval]
            F[sentence-transformers all-MiniLM-L6-v2]
            G[(FAISS L2 Flat Index)]
        end
        
        subgraph Inference Engine [Generation]
            H[llama.cpp AVX/SIMD]
            I[Llama-3.x Q8_0 GGUF]
        end
    end

    A -- JSON HTTP POST --> B
    A -- JSON HTTP POST --> C
    B --> D --> E
    E --> F --> G
    C --> F --> G
    G -- Top-K Context --> H
    H --> I
    I -- Grounded Answer --> A
'''

🚀 Key Engineering Features
1. Zero-Copy C++ Text Ingestion (pybind11)
Standard Python text chunking creates significant garbage collection churn by allocating new string objects for every sliding window slice. LocalMind mitigates this via a custom C++ backend extension. Using std::string_view, the engine calculates memory offset boundaries (size_t pairs) without intermediate heap allocations, lazily returning them to Python for a single slice operation.

2. Hardware-Aware Vector Retrieval
Bypassed heavy cloud vector databases in favor of Meta's faiss-cpu. The ingestion pipeline strictly enforces contiguous float32 C-order memory layouts for Python arrays, guaranteeing that the underlying C++ FAISS backend can utilize AVX/SIMD hardware intrinsics for sub-millisecond L2 Euclidean distance searches.

3. Quantized CPU Inference
To accommodate strict RAM limits (8GB - 16GB) without Out-Of-Memory (OOM) crashes, the system avoids standard FP32 weights. It deploys GGUF-quantized models (e.g., 8-bit or 4-bit integer weights) via llama-cpp-python, constraining the active LLM memory footprint to ~1.3GB – 2.5GB while maintaining high reasoning fidelity.

4. Transparent Retrieval Diagnostics
RAG systems frequently fail silently at the retrieval boundary. LocalMind's API payload includes the raw Top-K retrieved chunks and their exact L2 vector distances, exposed directly in the UI for immediate debugging of dense retrieval failures versus LLM reasoning limits.

📊 Performance Profiling & Empirical Benchmarks
A core objective was profiling the ABI boundary overhead between CPython and C++ extensions.

Experiment: Benchmarked the zero-copy C++ chunker against Python's native text[start:end] slicing on a 10MB simulated document.

Result: The C++ backend yielded a 1.01x speedup (0.0188s C++ vs. 0.0191s Python).

Architectural Insight: Empirical profiling proved that Python's native string slicing internally executes a highly optimized C function (PyUnicode_Substring). The microsecond optimizations gained by std::string_view were entirely offset by the pybind11 marshaling overhead of pushing integer pairs back into a Python list. This established a critical project guideline: C++ extensions in Python pipelines must be strictly reserved for SIMD-heavy mathematical vectorization (like FAISS/llama.cpp) or releasing the GIL, not scalar string manipulation.

🛠️ Build and Deployment Instructions
Prerequisites
Python 3.10+

MSVC / GCC (C++20 compatible compiler for the pybind11 extension)

1. Setup Environment
Bash
git clone [https://github.com/yourusername/LocalMind.git](https://github.com/yourusername/LocalMind.git)
cd LocalMind
python -m venv venv
# Windows: .\venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
2. Compile C++ Extension & Install Dependencies
Bash
pip install -e . --no-build-isolation
pip install -r requirements.txt
3. Download Local Inference Model
Bash
python scripts/download_model.py
4. Launch Microservices
Open two separate terminal instances:

Terminal 1: Inference API

Bash
uvicorn api:app --reload --port 8000
Terminal 2: Client UI

Bash
streamlit run app.py
Navigate to http://localhost:8501 to access the system.