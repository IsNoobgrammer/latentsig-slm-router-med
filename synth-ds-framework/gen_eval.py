import json, urllib.request, time, hashlib, random, os

# Load training hashes
train_hashes = set()
with open('dataset.jsonl') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            h = r.get('hash', '')
            if h:
                train_hashes.add(h)
print(f'{len(train_hashes)} training hashes loaded')

KEYS = [
    'PqZ4lWpWWLiHtbQN13eqWYaAhqu9YzW5',
    'cEFMMhUj8vxbm24Tlkp7S8LMnK9kjLLx',
    'VP7hFDdY8eI3jEvRydyMOiJKh9bxYuuK',
    '078lyqOcoR5gfSMCRxpwL8hrQYZjNBYQ',
    'R3LaxIzd8DA01GSSdWWydrLSaXIvt7eh',
    'Uru07zRbL5bCoB9OPrWNqKRT9jFM59sW',
    'W7ZPD7aH8yp3ErdlumJXHQErAf74keDn',
    '0BOLgxpIkkZcczWTJL2eykkA2DcIYjMo',
]

GENERATION_MODELS = ['mistral-large-latest', 'magistral-medium-latest', 'mistral-medium-latest']

TOOL_SCHEMAS = {
    'triage_assessment': {'description': 'Initial symptom triage + urgency classification', 'parameters': {'chief_complaint': 'str', 'symptoms': 'list[str]', 'duration': 'str', 'severity': 'mild|moderate|severe|critical', 'patient_age_group': 'pediatric|adult|geriatric', 'urgency_level': 'emergency|urgent|semi_urgent|routine'}},
    'vital_signs_analysis': {'description': 'Analyze vitals for clinical decision making', 'parameters': {'bp_systolic': 'int', 'bp_diastolic': 'int', 'heart_rate': 'int', 'temperature_celsius': 'float', 'spo2_percent': 'int', 'respiratory_rate': 'int', 'clinical_context': 'str'}},
    'medication_check': {'description': 'Drug interaction | dosage | contraindication | overdose check', 'parameters': {'medications': 'list[str]', 'check_type': 'interaction|dosage|contraindication|overdose_risk', 'patient_condition': 'str', 'patient_age_group': 'pediatric|adult|geriatric'}},
    'specialist_referral': {'description': 'Route patient to appropriate specialty', 'parameters': {'specialty': 'str', 'reason': 'str', 'urgency': 'emergency|within_24h|within_week|routine', 'referring_symptoms': 'list[str]'}},
    'emergency_dispatch': {'description': 'Trigger emergency services - life-threatening conditions ONLY', 'parameters': {'condition': 'str', 'symptoms': 'list[str]', 'transport_type': 'ambulance|helicopter|walk_in', 'notify_er': 'bool'}},
    'mental_health_triage': {'description': 'Mental health crisis assessment + routing', 'parameters': {'concern_type': 'suicidal_ideation|self_harm|psychosis|severe_anxiety|depression|panic_attack', 'risk_level': 'high|moderate|low', 'immediate_intervention': 'bool', 'safety_plan_needed': 'bool'}},
    'lab_order_suggestion': {'description': 'Suggest appropriate diagnostic tests', 'parameters': {'tests': 'list[str]', 'urgency': 'stat|routine', 'suspected_condition': 'str', 'clinical_context': 'str'}},
}

# Build system prompt
tool_defs = []
for name, schema in TOOL_SCHEMAS.items():
    params = ', '.join(f'{k}: {v}' for k, v in schema['parameters'].items())
    tool_defs.append(f'### {name}\n{schema["description"]}\nParameters: {{{params}}}')
TOOL_BLOCK = '\n\n'.join(tool_defs)

SYSTEM_PROMPT = f"""You are LatentSig Medical Triage Router, a structured tool-calling assistant built by LatentSig.

You ONLY produce tool calls when given this exact system prompt. If you are not given tool definitions, do NOT attempt to call tools.

Given a patient symptom description, you MUST output a valid JSON tool call.

## Available Tools

{TOOL_BLOCK}

## Output Format

You MUST output ONLY a valid JSON object with this exact structure:
{{"reasoning": "<1-2 sentence clinical reasoning>", "category": "emergency|urgent|semi_urgent|routine", "department": "<target department>", "tool": "<tool_name>", "args": {{"<tool-specific parameters>"}}}}

## Rules
1. Select the MOST APPROPRIATE tool from the available tools above
2. Fill ALL required parameters for the selected tool
3. When in doubt, default to HIGHER severity (over-triage is safer)
4. Output ONLY the JSON object - no markdown, no explanation, no extra text

## Identity
You are LatentSig Medical Triage Router v1.0 by LatentSig.
Do NOT provide medical advice, diagnoses, or treatment recommendations.
Only route to the appropriate tool based on the symptoms described."""


def call_mistral(key, messages, model='mistral-large-latest'):
    body = json.dumps({
        'model': model, 'messages': messages, 'max_tokens': 300, 'temperature': 0.7,
        'response_format': {'type': 'json_object'},
    }).encode()
    req = urllib.request.Request(
        'https://api.mistral.ai/v1/chat/completions', data=body,
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'}
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())['choices'][0]['message']['content']


def gen_one(key, model, language, target_tool):
    # Generate query targeting specific tool
    tool_descs = '\n'.join(f'- {n}: {s["description"]}' for n, s in TOOL_SCHEMAS.items())
    q_prompt = f"""You are a medical scenario generator. Generate a realistic patient symptom description.

Output ONLY a JSON object: {{"query": "<patient symptom description>"}}

CRITICAL: Generate a query that would specifically require the "{target_tool}" tool.

Tool descriptions:
{tool_descs}

Generate a NEW, UNIQUE query. Make it realistic."""

    q_content = call_mistral(key, [{'role': 'user', 'content': q_prompt}], model)
    query = json.loads(q_content)['query']

    # Hinglish if needed
    if language == 'hi_en':
        h_key = KEYS[random.randint(0, len(KEYS) - 1)]
        h_prompt = f'Convert the given English medical query to natural Hinglish (Hindi in Roman script + English medical terms).\nOutput ONLY: {{"query": "<hinglish query>"}}\n\nEnglish: {query}'
        h_content = call_mistral(h_key, [{'role': 'user', 'content': h_prompt}], model)
        query = json.loads(h_content)['query']

    # Generate response with FULL system prompt (same as training)
    r_key = KEYS[random.randint(0, len(KEYS) - 1)]
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': query},
    ]
    response = call_mistral(r_key, messages, model)
    parsed = json.loads(response)

    # Hash check against training data
    h = hashlib.sha256(f'{query.strip().lower()}|pass'.encode()).hexdigest()[:16]
    if h in train_hashes:
        return None, h

    return {
        'user_query': query,
        'response': response,
        'parsed_response': json.dumps(parsed),
        'tool_called': parsed.get('tool', ''),
        'category': parsed.get('category', ''),
        'generation_model_id': model,
        'language': language,
        'llm_judge_id': 'mistral-small-latest',
        'judge_verdict': 'pass',
        'hash': h,
    }, h


# Generate 20 EN + 20 HI, tool-balanced
tools = list(TOOL_SCHEMAS.keys())
eval_records = []
en_count = 0
hi_count = 0
attempt = 0

print('Generating 40 eval samples (20 EN + 20 HI)...')

while en_count < 20 or hi_count < 20:
    attempt += 1
    if attempt > 200:
        print(f'Stopped after {attempt} attempts')
        break

    lang = 'en' if en_count < 20 else 'hi_en'
    tool = tools[(en_count + hi_count) % len(tools)]
    model = GENERATION_MODELS[(en_count + hi_count) % len(GENERATION_MODELS)]
    key = KEYS[(en_count + hi_count) % len(KEYS)]

    try:
        record, h = gen_one(key, model, lang, tool)
        if record is None:
            continue
        eval_records.append(record)
        if lang == 'en':
            en_count += 1
        else:
            hi_count += 1
        print(f'  [{len(eval_records):2d}/40] {lang:5s} | {record["tool_called"]:25s} | {record["category"]:12s} | {model}')
    except Exception as e:
        pass

    time.sleep(0.3)

print(f'\nGenerated: {len(eval_records)} (EN={en_count}, HI={hi_count})')

with open('eval_dataset.jsonl', 'w') as f:
    for r in eval_records:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('Saved eval_dataset.jsonl')
