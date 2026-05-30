# Setup Guide — LatentSig Medical Triage Router

Inference and evaluation only. Training is done.

---

## Colab — GGUF Inference (Fastest)

### Cell 1: Install

```python
!pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu128 --no-cache-dir --force-reinstall
!git clone https://github.com/IsNoobgrammer/latentsig-slm-router-med.git -q
!pip install -e latentsig-slm-router-med -q
%cd latentsig-slm-router-med
```

### Cell 2: Download + Load

```python
from huggingface_hub import hf_hub_download
from src.inference import LlamaCppEngine
from src.prompts import SYSTEM_PROMPT

gguf_path = hf_hub_download(
    repo_id="fhai50032/latentsig-med-router-qwen3-4b-gguf",
    filename="qwen3-4b-instruct-2507.Q5_K_M.gguf",
)

engine = LlamaCppEngine(model_path=gguf_path)
engine.load()
```

### Cell 3: Test

```python
response, latency = engine.generate(SYSTEM_PROMPT, "chest pain, 55yo male")
print(f"{latency:.2f}s\n{response}")
```

### Cell 4: Eval (batched, 8 parallel)

```python
!python -m src.eval --mode gguf --gguf-path {gguf_path}
```

---

## Colab — Unsloth Inference (Best Accuracy)

Higher accuracy than GGUF. Auto-downloads adapter from Hub.

### Cell 1: Install

```python
!pip install unsloth datasets -q
!git clone https://github.com/IsNoobgrammer/latentsig-slm-router-med.git -q
!pip install -e latentsig-slm-router-med -q
%cd latentsig-slm-router-med
```

### Cell 2: Load (auto-downloads adapter)

```python
from src.inference import UnslothEngine
from src.prompts import SYSTEM_PROMPT

engine = UnslothEngine(
    base_model="unsloth/Qwen3-4B-Instruct",
    adapter_path="fhai50032/latentsig-med-router-qwen3-4b",
)
engine.load()
```

### Cell 3: Test

```python
response, latency = engine.generate(SYSTEM_PROMPT, "chest pain, 55yo male")
print(f"{latency:.2f}s\n{response}")
```

### Cell 4: Eval (batched, 4 parallel)

```python
!python -m src.eval --mode slm --adapter-path fhai50032/latentsig-med-router-qwen3-4b
```

### Cell 5: Compare vs Mistral

```python
!python -m src.eval --mode full --adapter-path fhai50032/latentsig-med-router-qwen3-4b
```

---

## Local — llama.cpp

```bash
# Install
brew install llama.cpp           # macOS
winget install llama.cpp         # Windows

# Run
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
    "max_tokens": 256
  }'
```

---

## Local — Python

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="fhai50032/latentsig-med-router-qwen3-4b-gguf",
    filename="qwen3-4b-instruct-2507.Q5_K_M.gguf",
    n_gpu_layers=-1,
)

output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are LatentSig Medical Triage Router. Output ONLY valid JSON."},
        {"role": "user", "content": "chest pain, 55yo male"},
    ],
    response_format={"type": "json_object"},
)
print(output["choices"][0]["message"]["content"])
```

---

## Ollama / Docker

```bash
ollama run hf.co/fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M
docker model run hf.co/fhai50032/latentsig-med-router-qwen3-4b-gguf:Q5_K_M
```

---

## Model Files

| File | Size | Engine | Speed | Accuracy |
|------|------|--------|-------|----------|
| `Q5_K_M.gguf` | 2.89 GB | llama.cpp | ~1-3s | Good |
| `Q8_0.gguf` | 4.28 GB | llama.cpp | ~2-4s | Better |
| Unsloth adapter | ~500MB | PyTorch | ~3-6s | Best |

Hub GGUF: https://huggingface.co/fhai50032/latentsig-med-router-qwen3-4b-gguf  
Hub Adapter: https://huggingface.co/fhai50032/latentsig-med-router-qwen3-4b

---

## Eval Reference

```bash
# GGUF (fast, batched 8x)
python -m src.eval --mode gguf --gguf-path ./model.gguf

# Unsloth (best accuracy, batched 4x)
python -m src.eval --mode slm --adapter-path fhai50032/latentsig-med-router-qwen3-4b

# Mistral baseline
python -m src.eval --mode mistral

# Full comparison (auto GGUF if --gguf-path, else Unsloth)
python -m src.eval --mode full --adapter-path fhai50032/latentsig-med-router-qwen3-4b
python -m src.eval --mode full --gguf-path ./model.gguf

# Custom batch size
python -m src.eval --mode slm --adapter-path ... --batch-size 2

# Quick test
python -m src.eval --mode gguf --gguf-path ./model.gguf --limit 10

# Interactive
python -m src.test_engine --engine gguf --gguf-path ./model.gguf --query "chest pain"
python -m src.test_engine --engine slm --query "chest pain"
```

---

## Troubleshooting

**OOM with Unsloth batch:** Lower batch size `--batch-size 2` or `--batch-size 1`

**llama-cpp-python CPU only:** Use direct wheel URL:
```bash
!pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.23-cu125/llama_cpp_python-0.3.23-py3-none-linux_x86_64.whl"
```

**ModuleNotFoundError: src:** Run `!pip install -e latentsig-slm-router-med`

**Adapter not found:** Engine auto-downloads from Hub. Check HF token if private repo.
