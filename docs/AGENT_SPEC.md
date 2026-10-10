# AGENT SPEC — chat agent, LLM providers, budget guard, medical extraction

## 1. Principles

1. **Retrieval does not need the LLM.** The LLM only translates/normalises the question and writes the answer. The
   Retrieval Lab, Benchmark and Graph pages work with no LLM at all.
2. **Bounded agency.** A fixed plan (normalise → link → retrieve → answer) plus at most **one** follow-up tool round
   chosen by the model. That gives predictable cost, latency, and logs.
3. **Evidence or nothing.** The answer may only use the numbered triples it was given, and must cite them.
4. **Money is guarded on the server.** Every live call goes through the budget ledger; the default cap is $0.05 total.

## 2. Providers

```python
class LLMProvider(Protocol):
    name: str
    def chat(self, messages: list[dict], *, json_mode: bool = False, tools: list[dict] | None = None,
             max_tokens: int = 512, stream: bool = False, purpose: str) -> LLMResponse | Iterator[LLMDelta]: ...
```

| Provider | Notes |
|---|---|
| `mock` (default) | Deterministic, no network, $0. See §6 |
| `deepseek` | OpenAI SDK with `base_url=https://api.deepseek.com`, model `deepseek-flash`. **Send `extra_body={"thinking": {"type": "disabled"}}`**: thinking is on by default and then `temperature` is ignored. Use `temperature=0`. JSON mode via `response_format={"type": "json_object"}` (the prompt must contain the word "json"). Usage reports `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` / `completion_tokens` |

Timeout 30 s, one retry on 5xx/timeout. No retry on 4xx.

## 3. Budget guard

- Ledger: `RUNTIME_DIR/llm_ledger.jsonl`, append-only. Each line is `{ts, provider, model, purpose, session_id, prompt_hit, prompt_miss, completion, usd, estimate_usd}`.
- On start, `spent = Σ usd`. The ledger survives restarts; deleting the file is the only reset, and it should be deliberate.
- Before a call: `estimate = in_tokens_est × P_miss + max_tokens × P_out`, where `in_tokens_est = ceil(chars / 2.5)` (conservative for Vietnamese). If `spent + estimate > LLM_BUDGET_USD`, raise `BudgetExceeded` and do not call.
- After a call: `usd = hit × P_hit + miss × P_miss + completion × P_out` (prices per 1M tokens from config: 0.006 / 0.30 / 1.20). Append it.
- Concurrency: one process-wide lock around check + call + append. This is a single-user demo, so simplicity wins.
- `GET /api/budget` exposes it. The UI meter shows spent / cap and turns amber at 80% and red at 100%.

Expected cost (DeepSeek peak prices):

| Call | Tokens in / out | USD |
|---|---|---|
| Normalise | ~900 / ~80 | ~$0.0004 |
| Answer (30 triples) | ~1,700 / ~300 | ~$0.0009 |
| Chat turn (2 calls) | | **~$0.0013** → ~35 turns per $0.05 |
| Extraction (one ~3,000-char note) | ~2,500 / ~1,200 | ~$0.0022 |

## 4. Chat turn

### 4.1 Call 1: normalise (JSON mode, max_tokens 200)

System prompt:

```
You prepare questions for a knowledge-graph retriever. Reply with json only:
{"lang": "vi" | "en", "question_en": string, "mentions_en": string[], "intent": "lookup" | "list" | "count" | "compare" | "chitchat"}
Rules:
- question_en: a faithful, short English version of the user's question. Do not answer it.
- mentions_en: 1-3 entity names exactly as an English knowledge graph would name them (proper nouns, titles,
  standard English clinical terms). Resolve pronouns from the conversation.
- Vietnamese medical terms become standard English terms ("tăng huyết áp" → "hypertension", "đái tháo đường type 2" → "type 2 diabetes").
- lang is the language of the user's latest message.
The knowledge graph is: {kg_description}.
```

User content: the last 2 turns of history plus the new message. If `intent == "chitchat"`, skip retrieval and answer briefly
without citations (still counted against the budget).

### 4.2 Link and retrieve (no LLM)

- `Linker.link(mentions, kg)`: full-text search per mention (CVTs never returned).
  - Case-insensitive **exact** name matches win, up to 2 per mention (homonyms such as the two "The Illusion" nodes), highest degree first. Several anchors are fine: ADR scores each one separately.
  - Otherwise take the top hit only when the second hit's score ≤ `LINK_MARGIN` (0.8) × the top score.
  - Other mentions go to `anchors.candidates` for the user to pick. At most 3 anchors in total.
- With no anchors at all, end the turn with a message asking the user to choose (the UI shows chips).
- `retrieve(question_en, anchors, mode, k)` → `RetrievalResult`. The first `PROMPT_TRIPLES` (default 30) become `T1..Tn`.

Triple rendering in the prompt: `[T7] The Illusion (m.0yrlqjm) —theater production › play→ The Illusion`

- Relation label = the last two segments of the relation name, `_` → space.
- CVT nodes render as `[event#<eid>]`.
- Medical nodes add the Vietnamese label: `Hypertension (tăng huyết áp)`.

### 4.3 Call 2: answer (streamed, max_tokens 600)

System prompt:

```
Answer the user's question using ONLY the evidence triples below. Each line is "[T#] head —relation→ tail".
- Write in {lang}. At most 5 sentences, or a short list.
- After every factual claim, cite the supporting triple ids, e.g. "… [T3][T7]".
- If the evidence does not contain the answer, say so plainly and say what is missing. You may call ONE tool to get
  more evidence: expand_entity(eid, relation_hint) or retrieve_subgraph(question_en, anchors).
- Never use facts that are not in the triples.
{medical_notice}
Evidence:
{triples}
```

`{medical_notice}` (kg = medical only): `Kết thúc câu trả lời bằng: "Thông tin chỉ phục vụ minh hoạ nghiên cứu, không phải tư vấn y khoa."`

Tools (OpenAI function format, offered only in this call):

```json
[{"type": "function", "function": {"name": "expand_entity", "description": "Get up to 30 triples around one entity",
   "parameters": {"type": "object", "properties": {"eid": {"type": "integer"}, "relation_hint": {"type": "string"}},
                  "required": ["eid"]}}},
 {"type": "function", "function": {"name": "retrieve_subgraph", "description": "Run the graph retriever again with other anchors",
   "parameters": {"type": "object", "properties": {"question_en": {"type": "string"},
                  "anchors": {"type": "array", "items": {"type": "integer"}, "maxItems": 3}}, "required": ["question_en", "anchors"]}}}]
```

If a tool is called:
1. Run it.
2. Append the new triples as `T{n+1}..`.
3. Make **call 3** without tools (max_tokens 600).

If the provider has no native tool calls, accept a whole-message JSON `{"tool": name, "arguments": {...}}` instead.

### 4.4 Post-processing

- Parse `[T#]` citations. Drop ids that don't exist and log a warning.
- Send `citations` with the cited triples. The UI links each chip to the triple and graph node.
- If the answer has no citation and intent ≠ chitchat, append a note: "(không có trích dẫn bằng chứng)".

### 4.5 Degraded answer (budget exceeded, LLM error, or `llm: mock`)

Built from the retrieval result without an LLM:
- Score the candidate entities that are not anchors and not CVTs by the sum of the scores of the triples they appear in (top-30 triples).
- List the top 3 with the triples they come from, in the user's language: "Ứng viên từ đồ thị: **X** [T1][T4], **Y** [T2]".
- `done.degraded` says why.

## 5. Medical extraction

### 5.1 Output schema (`Extraction`, validated with pydantic)

```json
{
  "doc_title": "Ca 1: tăng calci máu",
  "entities": [
    {"id": "p1", "name_en": "Patient (case 1)", "name_vi": "Bệnh nhân (ca 1)", "type": "Patient"},
    {"id": "e1", "name_en": "hypercalcemia", "name_vi": "tăng calci máu", "type": "Condition"}
  ],
  "triples": [
    {"h": "p1", "r": "medical.patient.diagnosis", "t": "e1", "assertion": "present",
     "evidence_vi": "Lý do nhập viện: đến khám vì tăng calci máu"}
  ]
}
```

- `type` ∈ {Patient, Condition, Symptom, Drug, DrugClass, Procedure, LabTest, LabResult, Value, BodyPart, Finding}.
- `assertion` ∈ {present, negated, historical, family}.
- Limits per document: ≤ 60 entities, ≤ 120 triples.

### 5.2 Relation vocabulary (closed set)

Dotted, Freebase-like names, so WL's lexical relation-topic feature (which splits on `.` and `_`) can match question words:

| Relation | Head → tail |
|---|---|
| `medical.patient.diagnosis` | Patient → Condition |
| `medical.patient.history_of` | Patient → Condition (assertion historical) |
| `medical.patient.family_history` | Patient → Condition (assertion family) |
| `medical.patient.symptom` | Patient → Symptom |
| `medical.patient.denied_symptom` | Patient → Symptom (assertion negated) |
| `medical.patient.medication` | Patient → Drug |
| `medical.patient.procedure` | Patient → Procedure |
| `medical.patient.lab_result` | Patient → LabResult (n-ary node, like a Freebase CVT) |
| `medical.lab_result.test` | LabResult → LabTest |
| `medical.lab_result.value` | LabResult → Value (e.g. "12.0 mg/dL") |
| `medical.lab_result.interpretation` | LabResult → Finding (e.g. "elevated") |
| `medical.condition.symptom` | Condition → Symptom |
| `medical.condition.treated_with` | Condition → Drug \| Procedure |
| `medical.condition.complication` | Condition → Condition |
| `medical.condition.location` | Condition → BodyPart |
| `medical.drug.drug_class` | Drug → DrugClass |

### 5.3 Extraction prompt (JSON mode, max_tokens 1,500, temperature 0)

```
You extract a clinical knowledge graph from a Vietnamese clinical note. Reply with json only, matching this schema:
{schema}
Rules:
- One Patient entity per note (id "p1").
- name_en: the standard English clinical term (lowercase except proper names); name_vi: the term as written in the note.
- Use ONLY these relations: {relation table}. Map assertions: negated symptom → medical.patient.denied_symptom;
  past disease → medical.patient.history_of; relatives' disease → medical.patient.family_history.
- Each lab result is its own LabResult entity linked to its LabTest, Value and Finding.
- evidence_vi: the shortest quote from the note that supports the triple.
- Do not invent facts. Skip anything you cannot ground in the text.
```

Notes longer than 3,000 characters are split on section headings into ≤ 2 chunks, extracted separately, and merged by
`(type, lower(name_en))`. Invalid JSON gets one repair call ("Fix this to match the schema, json only"); after that, return 422.

### 5.4 Commit rules

- Node keys:
  - Shared concepts (Condition, Symptom, Drug, DrugClass, Procedure, LabTest, BodyPart, Finding) use `med:{type}:{slug(name_en)}`, so two notes share "hypertension".
  - Per-document nodes (Patient, LabResult, Value) use `med:{sha8(text)}:{id}`.
- Existing keys are reused (MERGE on `key`); new nodes get `eid = max(eid) + 1 …`.
- New relations get `rid = max(rid) + 1 …` and a `:Relation` node.
- New names and relations are GTE-embedded in batches of 32, appended to the overlay memmaps, and `emb_row` points into the overlay (offset by the bundle table size).
- Nothing in the bundle files changes.

## 6. Mock provider (deterministic)

| Purpose | Behaviour |
|---|---|
| normalise | If the message equals a benchmark question (case-insensitive), return that question and its `topic_entities` as mentions. Otherwise `lang` = "vi" when Vietnamese diacritics are present, `question_en` = the message unchanged, and `mentions_en` = quoted spans, else capitalised spans (≤ 3). For `kg=medical`, map terms through the sample fixtures' `name_vi → name_en` table |
| answer | The degraded answer of §4.5 (marked `degraded: "mock"`) |
| extract | If `sha256(text)` matches a sample, return its `*.extraction.json`; otherwise 422 `MOCK_CANNOT_EXTRACT` with a hint to enable live mode |

Every agent and ingest test runs on the mock. A single live smoke test (`make smoke-live`, ≤ $0.01, manual) checks the
DeepSeek path: one normalise, one answer, one extraction of the shortest sample.
