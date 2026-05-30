# Setup Guide — LatentSig Medical Triage Router

Inference and evaluation only. Training is done.

---

## Colab — Quick Start

### Cell 1: Install

```python
# Pre-built CUDA 12 wheel (direct URL — guaranteed GPU support)
!pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.23-cu125/llama_cpp_python-0.3.23-py3-none-linux_x86_64.whl" --no-cache-dir --force-reinstall
!git clone https://github.com/IsNoobgrammer/latentsig-slm-router-med.git -q
!pip install -e latentsig-slm-router-med -q
%cd latentsig-slm-router-med
```

### Cell 2: Download GGUF

```python
from huggingface_hub import hf_hub_download

gguf_path = hf_hub_download(
    repo_id="fhai50032/latentsig-med-router-qwen3-4b-gguf",
    filename="qwen3-4b-instruct-2507.Q5_K_M.gguf",
)
print(f"Model: {gguf_path}")
```

### Cell 3: Load + Test

```python
from src.inference import LlamaCppEngine
from src.prompts import SYSTEM_PROMPT

engine = LlamaCppEngine(model_path=gguf_path)
engine.load()

response, latency = engine.generate(
    SYSTEM_PROMPT,
    "68-year-old male, sudden facial droop, cannot speak, right arm weakness.",
)
print(f"Latency: {latency:.2f}s")
print(response)
```

### Cell 4: Run Eval

```python
!python -m src.eval --mode gguf --gguf-path {gguf_path}
```

### Cell 5: Interactive Test

```python
!python -m src.test_engine --engine gguf --gguf-path {gguf_path} --query "chest pain, 55yo male"
```

### Cell 6: Compare vs Mistral Baseline

```python
!python -m src.eval --mode full --gguf-path {gguf_path} --mistral-model mistral-small-latest
```

---

## Local — llama.cpp (Fastest)

### Install

```bash
# macOS
brew install llama.cpp

# Windows
winget install llama.cpp

# Or build from source
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && cmake -B build && cmake --build build -j
```

### Run Inference

```bash
# Download + run (auto-downloads from Hub)
llama-cli -hf fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M

# OpenAI-compatible server
llama-server -hf fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M
```

### Query the Server

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are LatentSig Medical Triage Router. Given a patient symptom description, output ONLY a valid JSON tool call with: reasoning, category, department, tool, args."},
      {"role": "user", "content": "68yo male, sudden facial droop, cannot speak."}
    ],
    "response_format": {"type": "json_object"},
    "max_tokens": 256,
    "temperature": 0.1
  }'
```

---

## Python — llama-cpp-python

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="fhai50032/latentsig-med-router-qwen3-4b-gguf",
    filename="qwen3-4b-instruct-2507.Q5_K_M.gguf",
    n_ctx=2048,
    n_gpu_layers=-1,  # offload all to GPU
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are LatentSig Medical Triage Router. Given a patient symptom description, output ONLY a valid JSON tool call."},
        {"role": "user", "content": "Crushing chest pain radiating to left arm, sweating, 55yo male."},
    ],
    response_format={"type": "json_object"},
    max_tokens=256,
    temperature=0.1,
)

print(output["choices"][0]["message"]["content"])
```

---

## Ollama

```bash
ollama run hf.co/fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M
```

---

## Docker

```bash
docker model run hf.co/fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M
```

---

## Model Files

| File | Size | Speed | Quality |
|------|------|-------|---------|
| `qwen3-4b-instruct-2507.Q5_K_M.gguf` | 2.89 GB | Fast | Good |
| `qwen3-4b-instruct-2507.Q8_0.gguf` | 4.28 GB | Medium | Best |

Hub: https://huggingface.co/fhai50032/latentsig-med-router-qwen3-4b-gguf

---

## Eval Reference

```bash
# Placeholder (pipeline test, no API)
python -m src.eval --mode placeholder

# Mistral baseline only
python -m src.eval --mode mistral

# GGUF (our fine-tuned model)
python -m src.eval --mode gguf --gguf-path ./model.gguf

# Full comparison (GGUF vs Mistral)
python -m src.eval --mode full --gguf-path ./model.gguf

# Full agent loop (end-to-end, slower)
python -m src.eval --mode gguf --gguf-path ./model.gguf --use-agent

# Limit samples (quick test)
python -m src.eval --mode gguf --gguf-path ./model.gguf --limit 10
```

---

## Troubleshooting

**llama-cpp-python install fails with CUDA:**
```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Out of memory:**
Use Q5_K_M (2.89 GB) — fits in any GPU with 4GB+ VRAM.

**Slow inference:**
Ensure `n_gpu_layers=-1` to offload all layers to GPU. CPU-only is 5-10x slower.
