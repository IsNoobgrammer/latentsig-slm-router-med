     1|# LatentSig Medical Triage Router
     2|
     3|> A local-first agentic workflow where a fine-tuned Small Language Model (SLM) acts as a reliable structured router for clinical medical triage.
     4|
     5|**Author:** LatentSig  
     6|**Model:** Qwen3-4B-Instruct (QLoRA fine-tuned)  
     7|**Dataset:** [fhai50032/latentsig-med-triage-router](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router)  
     8|**GitHub:** [IsNoobgrammer/latentsig-slm-router-med](https://github.com/IsNoobgrammer/latentsig-slm-router-med)
     9|
    10|---
    11|
    12|## Architecture Overview
    13|
    14|![Architecture Overview](visuals/architecture_overview.png)
    15|
    16|### Agent Pipeline
    17|
    18|![Agent Pipeline](visuals/agent_pipeline.png)
    19|
    20|The SLM runs twice per query:
    21|1. **Tool Call** — reads tool definitions, selects tool, outputs JSON
    22|2. **Response** — reads tool result, synthesizes human-readable summary
    23|
    24|---
    25|
    26|## Project Structure
    27|
    28|```
    29|latentsig-slm-router-med/
    30|│
    31|├── src/                              ← Agentic loop + inference
    32|│   ├── agent.py                      ← Phase 2: ReAct loop (placeholder engine)
    33|│   ├── agent_colab.py                ← Phase 2: Two-stage SLM loop (Colab)
    34|│   ├── config.py                     ← Model paths, hyperparams
    35|│   ├── inference.py                  ← SLMEngine (Unsloth) + PlaceholderEngine
    36|│   ├── parser.py                     ← JSON extraction + Pydantic validation
    37|│   ├── prompts.py                    ← System prompts (tool-call + assistant)
    38|│   ├── tools.py                      ← 7 deterministic mock tools + CSV logging
    39|│   └── test_agent.py                 ← Smoke test (10 queries)
    40|│
    41|├── synth-ds-framework/               ← Dataset generation pipeline
    42|│   ├── orchestrator_parallel.py      ← Parallel datagen (32 workers, 8 keys)
    43|│   ├── orchestrator.py               ← Serial datagen (fallback)
    44|│   ├── verifier.py                   ← 3-layer: Pydantic + tool enforce + LLM judge
    45|│   ├── tool_schemas.py               ← 7 tool definitions (flat params)
    46|│   ├── prompts.py                    ← Query generation prompts
    47|│   ├── models.py                     ← Pydantic models (TriageCall, DatasetRecord)
    48|│   ├── monitor.py                    ← Flask dashboard (localhost:5000)
    49|│   ├── gen_eval.py                   ← Eval dataset generator
    50|│   ├── gen_visuals.py                ← Seaborn/matplotlib visualizations
    51|│   ├── gen_flowcharts.py             ← Flowchart image generator
    52|│   └── tool-schema/                  ← 91+ extended tool schemas (future)
    53|│
    54|├── tool_logs/                        ← Per-tool CSV logs (generated at runtime)
    55|│   ├── triage_assessment.csv
    56|│   ├── vital_signs_analysis.csv
    57|│   ├── medication_check.csv
    58|│   ├── specialist_referral.csv
    59|│   ├── emergency_dispatch.csv
    60|│   ├── mental_health_triage.csv
    61|│   └── lab_order_suggestion.csv
    62|│
    63|└── README.md                         ← This file
    64|```
    65|
    66|---
    67|
    68|## Tool Definitions
    69|
    70|The SLM must select from 7 tools. Each tool has flat parameters (no nesting) for easier SLM learning.
    71|
    72|| Tool | Description | Key Parameters |
    73||------|-------------|----------------|
    74|| `triage_assessment` | Initial symptom triage + urgency | chief_complaint, symptoms, severity, urgency_level |
    75|| `vital_signs_analysis` | Analyze vitals for clinical decisions | bp_systolic, bp_diastolic, heart_rate, spo2_percent |
    76|| `medication_check` | Drug interaction / dosage / overdose | medications, check_type, patient_condition |
    77|| `specialist_referral` | Route to appropriate specialty | specialty, reason, urgency |
    78|| `emergency_dispatch` | Life-threatening conditions ONLY | condition, symptoms, transport_type |
    79|| `mental_health_triage` | Crisis assessment + routing | concern_type, risk_level, immediate_intervention |
    80|| `lab_order_suggestion` | Suggest diagnostic tests | tests, urgency, suspected_condition |
    81|
    82|### Output Format
    83|
    84|```json
    85|{
    86|  "reasoning": "1-2 sentence clinical reasoning",
    87|  "category": "emergency|urgent|semi_urgent|routine",
    88|  "department": "<target department>",
    89|  "tool": "<tool_name>",
    90|  "args": { /* tool-specific parameters */ }
    91|}
    92|```
    93|
    94|---
    95|
    96|## Dataset
    97|
    98|**HF:** [fhai50032/latentsig-med-triage-router](https://huggingface.co/datasets/fhai50032/latentsig-med-triage-router)
    99|
   100|| Split | Samples | Languages |
   101||-------|---------|-----------|
   102|| Train | 2,000 | 1,000 EN + 1,000 Hinglish |
   103|| Eval | 40 | 20 EN + 20 Hinglish |
   104|
   105|### Generation Pipeline
   106|
   107|```
   108|┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   109|│  Tool Schema    │───▶│  Query Generator  │───▶│  Response Gen   │
   110|│  (7 tools)      │    │  (3 Mistral models)│   │  (with hint)    │
   111|└─────────────────┘    └──────────────────┘    └─────────────────┘
   112|                                                        │
   113|                                                        ▼
   114|┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
   115|│  Dataset JSONL  │◀───│  3-Layer Verify   │◀───│  Target Tool    │
   116|│  (2,000 rows)   │    │  (Pydantic+Fuzzy  │   │  Enforcement    │
   117|│                 │    │   +LLM Judge)     │   │                 │
   118|└─────────────────┘    └──────────────────┘    └─────────────────┘
   119|```
   120|
   121|**Key design:**
   122|- **Tool-balanced:** `pick_target_tool()` uses weighted random favoring least-used tools
   123|- **Target hint:** Response prompt includes `[Use the X tool]` for clean training data
   124|- **3-layer verification:** Pydantic structural → tool enforcement → LLM judge (unbiased)
   125|- **Hash dedup:** `sha256(user_query + verdict)` — no duplicate queries
   126|- **Parallel:** 32 workers, 8 Mistral API keys, ~0.7/s throughput
   127|
   128|### Generation Models
   129|
   130|| Model | Purpose | Share |
   131||-------|---------|-------|
   132|| mistral-large-latest | Query + response generation | 23% |
   133|| mistral-medium-latest | Query + response generation | 50% |
   134|| magistral-medium-latest | Query + response generation | 27% |
   135|| mistral-small-latest | LLM judge (unbiased) | 100% |
   136|
   137|---
   138|
   139|## Agentic Loop (Phase 2)
   140|
   141|### ReAct Loop Flow
   142|
   143|![ReAct Loop Flow](visuals/react_loop_flow.png)
   144|
   145|### Hallucination Recovery
   146|
   147|![Hallucination Recovery](visuals/hallucination_recovery.png)
   148|
   149|### Two-Stage Inference (agent_colab.py)
   150|
   151|```python
   152|# Stage 1: Tool Call
   153|raw, latency = engine.generate(TOOL_CALL_SYSTEM_PROMPT, query)
   154|tool_call = parse_tool_call(raw)  # JSON extraction + validation
   155|tool_result = execute_tool(tool_call["tool"], tool_call["args"])
   156|db.log(tool_call["tool"], tool_call["args"], tool_result)
   157|
   158|# Stage 2: Response
   159|context = f"Query: {query}\nDecision: {tool_call}\nResult: {tool_result}"
   160|response, latency = engine.generate(ASSISTANT_SYSTEM_PROMPT, context)
   161|```
   162|
   163|### Mock Tools (Deterministic)
   164|
   165|Each tool returns fixed data for the same input. All calls logged to CSV.
   166|
   167|```python
   168|# Example: emergency_dispatch
   169|def emergency_dispatch(args):
   170|    return {
   171|        "status": "dispatched",
   172|        "condition": args["condition"],
   173|        "transport": args["transport_type"],
   174|        "eta_minutes": 8,
   175|        "dispatch_id": f"EMD-{hash(str(args)) % 10**8:08d}"
   176|    }
   177|```
   178|
   179|**Log output:** `tool_logs/emergency_dispatch.csv`
   180|```
   181|call_id,tool,args,result,timestamp
   182|EMD-59F761B4,emergency_dispatch,{...},{...},2026-05-29T18:57:01
   183|```
   184|
   185|---
   186|
   187|## Fine-Tuning
   188|
   189|**Model:** Qwen3-4B-Instruct | **Method:** QLoRA 4-bit | **Hardware:** Colab T4 (16GB)
   190|
   191|### Training Config
   192|
   193|```python
   194|SFTConfig(
   195|    dataset_text_field="text",
   196|    per_device_train_batch_size=16,
   197|    gradient_accumulation_steps=1,
   198|    num_train_epochs=2,
   199|    learning_rate=7e-5,
   200|    optim="adamw_8bit",
   201|    fp16=True,
   202|    eval_strategy="steps",
   203|    eval_steps=25,
   204|    save_strategy="steps",
   205|    save_steps=50,
   206|    logging_steps=1,
   207|    report_to="wandb",
   208|)
   209|```
   210|
   211|### LoRA Config
   212|
   213|```python
   214|FastLanguageModel.get_peft_model(
   215|    model,
   216|    r=16,
   217|    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
   218|                     "gate_proj", "up_proj", "down_proj"],
   219|    lora_alpha=32,
   220|    lora_dropout=0,
   221|    use_gradient_checkpointing="unsloth",
   222|)
   223|```
   224|
   225|### Dataset Format (Qwen3 Chat Template)
   226|
   227|```python
   228|def format_to_text(row):
   229|    messages = [
   230|        {"role": "system", "content": SYSTEM_PROMPT},
   231|        {"role": "user", "content": row["user_query"]},
   232|        {"role": "assistant", "content": row["response"]},
   233|    ]
   234|    return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
   235|
   236|ds = ds.map(format_to_text, remove_columns=ds.column_names)
   237|```
   238|
   239|---
   240|
   241|## Verification Pipeline
   242|
   243|Every sample passes 3 layers before inclusion:
   244|
   245|![Verification Pipeline](visuals/verification_pipeline.png)
   246|
   247|**Critical:** The LLM judge does NOT know which tool was targeted. It evaluates tool selection purely on medical merit.
   248|
   249|---
   250|
   251|## Monitoring
   252|
   253|**Dashboard:** `python synth-ds-framework/monitor.py` → http://localhost:5000
   254|
   255|- Live progress bars (EN / HI_EN vs target)
   256|- Stats cards: total, passed, failed, rate, ETA
   257|- Category / tool / model breakdowns
   258|- Recent samples table
   259|- Dataset viewer at /viewer (sortable, filterable)
   260|
   261|---
   262|
   263|## Quick Start
   264|
   265|### 1. Generate Dataset (already done)
   266|
   267|```bash
   268|cd synth-ds-framework
   269|python orchestrator_parallel.py --en 1000 --hi-en 1000 --workers 32
   270|```
   271|
   272|### 2. Fine-Tune (Colab)
   273|
   274|```python
   275|# Load model
   276|from unsloth import FastLanguageModel
   277|model, tokenizer = FastLanguageModel.from_pretrained(
   278|    model_name="unsloth/Qwen3-4B-Instruct",
   279|    max_seq_length=1280, load_in_4bit=True,
   280|)
   281|
   282|# Load dataset
   283|from datasets import load_dataset
   284|ds = load_dataset("fhai50032/latentsig-med-triage-router", split="train")
   285|eval_ds = load_dataset("fhai50032/latentsig-med-triage-router", split="eval")
   286|
   287|# Format + train (see finetune.md for full config)
   288|```
   289|
   290|### 3. Run Agent
   291|
   292|```python
   293|from src.agent_colab import SLMEngine, TriageAgent, ToolDB
   294|
   295|engine = SLMEngine(adapter_path="fhai50032/latentsig-med-router-qwen3-4b")
   296|engine.load()
   297|
   298|agent = TriageAgent(engine)
   299|
   300|# Verbose: shows full ReAct loop (tool call, execution, response)
   301|result = agent.run("68-year-old male, sudden facial droop, cannot speak", verbose=True)
   302|
   303|# Silent: only returns result object
   304|result = agent.run("chest pain, 55yo male", verbose=False)
   305|
   306|print(result.response)          # Human-readable answer
   307|print(result.tool_call)         # JSON tool call
   308|print(result.tool_result)       # Deterministic tool output
   309|print(result.total_latency_ms)  # End-to-end latency
   310|```
   311|
   312|---
   313|
   314|## Key Design Decisions
   315|
   316|| Decision | Choice | Rationale |
   317||----------|--------|-----------|
   318|| SLM | Qwen3-4B-Instruct | #1 fine-tuning benchmark, strong multilingual, Apache 2.0 |
   319|| Fine-tuning | QLoRA 4-bit | Fits T4 16GB, ~5-6GB VRAM |
   320|| LoRA rank | r=16 | Good capacity/speed balance for 4B |
   321|| System prompt | Tool defs in prompt | Model learns to READ tools, not memorize |
   322|| Identity | LatentSig branded | Only tool-calls when given this specific prompt |
   323|| Tools | 7 flat-param tools | Easier for 1B-3B models to learn |
   324|| Verification | 3-layer (Pydantic+Tool+Judge) | Fast structural + unbiased accuracy |
   325|| Generation | 3 Mistral models rotated | Diversity in training data |
   326|| Dedup | Hash on save only | Failed samples can retry same query |
   327|| Output | Per-tool CSV logs | Verifiable out-of-loop |
   328|
   329|---
   330|
   331|## License
   332|
   333|MIT — LatentSig, 2026
   334|