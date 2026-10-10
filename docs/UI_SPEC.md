# UI SPEC — FastGraphRetriever web app

Build with the vendored skills: load `.claude/skills/frontend-design` before any visual work, and use
`.claude/skills/webapp-testing` for screenshot reviews. Optional extra reference on the dev machine (not vendored, review it
before installing): `nextlevelbuilder/ui-ux-pro-max-skill` pinned at `50d8a7de0900119855614541f15a1a616691eb33`
(MIT; 73 files incl. Python search scripts) → copy `.claude/skills/ui-ux-pro-max/` into `~/.claude/skills/`.

## 1. Brief

- **Subject:** evidence retrieval over a knowledge graph. The research method finds *bridges* (AER) and handles *several anchors* (ADR); the medical use case is Vietnamese clinical notes.
- **Audience:** examiners and colleagues watching a live demo on a projector or a laptop, and a researcher driving it.
- **Primary job:** make it obvious *which* evidence was retrieved and *why*, fast enough that nobody waits.
- **Voice:** precise, calm, clinical. Vietnamese first.

## 2. Design plan (tokens)

### 2.1 Colour: "formalin & iodine"

The palette comes from the histology bench: stains that make structure visible, which is what the retriever does for a graph.

| Token | Light | Dark | Role |
|---|---|---|---|
| `--ground` | `#EDF1EF` (celadon grey) | `#0F1B1E` (slate-teal) | Page background |
| `--ink` | `#16213A` (navy ink) | `#E2EBE7` | Text, default nodes and edges |
| `--formalin` | `#0F766E` | `#2DD4BF` | Primary actions; **anchor** entities; base pass |
| `--iodine` | `#B45309` | `#F59E0B` | **Bridge** entities and bridge passes (AER) |
| `--methylene` | `#6D28D9` | `#A78BFA` | **Answers** (gold in benchmark, cited candidates in chat); per-anchor passes (ADR) |
| `--signal` | `#B91C1C` | `#F87171` | Errors, budget exhausted |

- Neutrals derive from `--ink` at 8/16/32/64% alpha on `--ground`, so surfaces are tints, not new hues.
- Contrast: text tokens ≥ 4.5:1 on `--ground` in both themes. Verify with a contrast check in M7.
- Colour never carries meaning alone (§4.3 adds shapes).

### 2.2 Type

- **Be Vietnam Pro** (Google Fonts; a Vietnamese foundry, complete diacritics) for everything readable: 400/500/600, scale 13 / 15 / 18 / 24 / 32 px, line-height 1.5 for body and 1.25 for headings, body line length ≤ 72ch.
- **JetBrains Mono**, only for genuine identifiers: relation ids (`theater.theater_production.play`), eids, MIDs, timings. Not for labels.
- Headings are sentence case. No all-caps labels, no eyebrow labels above headings.

### 2.3 Layout concept: evidence beside the conversation

```
┌ rail ┬──────────── conversation (≤72ch) ────────────┬────────────── evidence workspace ──────────────┐
│ ◆    │ Hỏi: Vở kịch nào do The Illusion sản xuất?    │  [ graph canvas: anchors ◆, bridges ⬢, answers ◎ ] │
│ Chat │                                               │                                                │
│ Lab  │ Đáp: The Illusion (vở kịch) [T1]. …           │                                                │
│ Bench│   ↳ chips: [T1] [T3]                          │ ── retrieval trace ───────────────────────────── │
│ Nhập │                                               │  base      ▮▮▮▮▮▮▮▮▮▮▮▮░░░  12 ms               │
│ Đồ thị│                                              │  + bridges ▮▮▮░▮▮▮▮░░      11 ms   +18 triples   │
│      │ [ composer ……………………………… ] (Gửi)              │  ── triples ── #  head  relation  tail  score  │
└──────┴───────────────────────────────────────────────┴────────────────────────────────────────────────┘
  header strip (thin): graph [GrailQA ▾] · mode [Base|ADR|AER|ADR+AER] · k [100] · budget ▬▬▬░ $0.012/$0.05 · status ●
```

- Left-aligned text throughout. The evidence workspace takes 58% of the width at ≥ 1280 px.
- 768–1279 px: evidence becomes a right drawer, toggled from the answer.
- < 768 px: tabs "Trả lời / Bằng chứng".

### 2.4 Principles

1. **The trace is the hero.** One memorable element: the **Retrieval Trace** (§3.1). Everything around it stays quiet.
2. **Show what changed.** Motion only answers an action: a pass lane fills as its triples arrive, and graph nodes fade in by pass. No idle animation; respect `prefers-reduced-motion` (instant states).
3. **Evidence is clickable everywhere.** A citation chip, triple row, graph node and trace tick for the same triple highlight each other.
4. **Plain words.** "Cầu nối" (bridge), "Điểm neo" (anchor), "Lượt" (pass). The method names (ADR, AER) appear once, with a tooltip.

Review this plan against the brief before building (frontend-design process). If a part reads like a default you'd give any
dashboard (card soup, gradient washes, a big-number hero), revise it and note why in the PR.

## 3. Signature components

### 3.1 Retrieval Trace

- One lane per pass, in execution order. The sequence is real, so numbering is fine.
- Lane label: kind (Base / Cầu nối từ {anchors} / Neo riêng: {anchor}) + ms + "+N mới" (triples in the final top-k whose `best_pass` is this lane).
- Inside each lane: ticks at the ranks (1..k) of the triples won by that lane, coloured by kind (base = formalin, bridge = iodine, single = methylene).
- Hovering a lane dims the triples and graph nodes it didn't contribute. Clicking a tick selects the triple.
- Footer: a stacked latency bar (encode · link · fetch · features · prepare · passes · fuse) with ms per segment.

### 3.2 Graph canvas

- Cytoscape.js + `cytoscape-fcose`; at most 150 nodes drawn (top-k triples' endpoints, anchors pinned left).
- Node encodings (shape + colour):

| Kind | Shape | Colour |
|---|---|---|
| Anchor | diamond | formalin |
| Bridge | hexagon | iodine |
| Answer / cited candidate | ring | methylene |
| CVT | small hollow circle | ink 32% |
| Other | dot | ink 64% |

- Edge width encodes rank (top-10 thicker). Edge labels show only on hover/selection (humanised relation).
- Controls: fit, layout re-run, "chỉ hiện top 20/50/all", export PNG.

### 3.3 Triple table

- Virtualised. Columns: # · head · relation (mono id + humanised label) · tail · score · lượt.
- Filter by entity and by pass. Gold-answer rows marked in benchmark mode.

### 3.4 Other components

ModeSwitch (segmented, 4 options, tooltips explaining each mode in one sentence) · KSelect · KGSwitch · BudgetMeter
(spent/cap bar, amber ≥ 80%, signal at 100%, tooltip with call count) · AnchorPicker (search-as-you-type over `/entities/search`,
chips, removable) · CitationChip · StatusDot (`/health`) · Notice (truncated subgraph, degraded answer, medical disclaimer).

## 4. Pages

### 4.1 Chat (`/`)

- Composer placeholder: "Hỏi về đồ thị tri thức… (tiếng Việt hoặc tiếng Anh)".
- While a turn runs, the status line follows the SSE `status` stages ("Đang chuẩn hoá câu hỏi" → "Đang liên kết thực thể" → "Đang truy xuất" → "Đang trả lời").
- The `plan` event shows a collapsible "Câu hỏi đã chuẩn hoá: …" with the English question and mentions, so the VI→EN step is visible.
- Anchor chips appear above the answer and can be edited; changing them re-runs retrieval and answer.
- Degraded answers carry a quiet notice explaining why (budget, LLM error, mock).
- Empty state: 4 example questions (2 GrailQA, 2 medical once ingested) as buttons.

### 4.2 Retrieval Lab (`/lab`)

- An English question, AnchorPicker, mode, k, and source (Neo4j / benchmark id).
- "So sánh" splits the workspace into two traces + graphs (e.g. Base vs ADR+AER) with a diff list (gained / lost / moved).
- This is the page used to explain the method live. It never calls the LLM.

### 4.3 Benchmark (`/benchmark`)

- Left: a filterable question table (hops, level, function, domain, search).
- Right: the selected question run in any mode, with gold answers ringed and ansR@50/100/200 shown.
- Aggregate tab: run N questions (linspace sample) for the chosen modes, with SSE progress.
- Chart: **ansR vs k** (50/100/200), lines for the live modes plus the reference baselines (SubgraphRAG, REL-RAG, dashed). Follow the dataviz skill if available. Direct labels, no legend box.
- A footnote states the ODA-dev caveat.

### 4.4 Nhập liệu y khoa (`/ingest`)

- Sample picker (3 synthetic notes) or a text area (≤ 6,000 chars, counter).
- "Trích xuất" shows the estimated cost and the live/mock badge before running.
- Preview: entity table (EN / VI / type, editable) and triple table (with the evidence quote highlighted in the note text).
- "Thêm vào đồ thị" commits; "Xoá đồ thị y khoa" resets (confirm dialog naming the counts).
- A permanent notice: "Dữ liệu mẫu là giả lập. Ứng dụng phục vụ nghiên cứu, không dùng cho chẩn đoán."

### 4.5 Đồ thị (`/explore`)

- Entity search → 1-hop neighbourhood. Click a node to expand (≤ 50 triples per expansion).
- "Dùng làm điểm neo" sends the node to the Lab.

## 5. Copy rules

- Vietnamese default, English toggle (i18next; keys in `web/src/i18n/{vi,en}.json`; no hard-coded strings).
- Buttons name the action: "Truy xuất", "So sánh", "Trích xuất", "Thêm vào đồ thị". Toasts reuse the verb ("Đã thêm 24 thực thể").
- Errors say what happened and what to do: "Không kết nối được Neo4j. Kiểm tra container `neo4j` rồi tải lại trang."

## 6. Tech

Vite + React 18 + TypeScript (strict) · Tailwind CSS 3.4 with the tokens as CSS variables · Radix UI primitives (Tabs, ToggleGroup,
Tooltip, Dialog, Popover) · TanStack Query (REST) + a small SSE client (`fetch` + `ReadableStream`, so it works with POST) · Zustand for UI
state · Cytoscape.js 3 + cytoscape-fcose · Recharts for the benchmark chart · react-markdown + rehype-sanitize · i18next ·
lucide-react icons. Fonts self-hosted via `@fontsource` (no runtime Google calls).

Quality floor: keyboard-reachable controls with a visible focus ring; `aria-live` for streaming status; graph has a text
alternative (the triple table); respects reduced motion; works at 1280×720 (projector) and 390 px wide.

## 7. Acceptance (M4/M5/M7)

- Playwright smoke (webapp-testing skill): each page loads with no console errors, in light and dark, at 1280×720 and 390×844. Screenshots saved under `web/test-results/`.
- Retrieval Lab: Base vs ADR+AER on benchmark question `2101960008000` shows both traces, and the diff lists gained/lost triples.
- Chat (mock): a benchmark question returns a degraded answer with ≥ 1 citation chip, and clicking it highlights the same triple in the table and the graph.
- Benchmark aggregate for 30 linspace questions matches `make parity` numbers.
