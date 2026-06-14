# Learning Notes — Context Engineering & the Agent Loop

Notes built up while reading through this demo. The running example is the
*ACV Helios-IX* starship ops manual: a Claude agent answers questions about it
using three tools over a BM25-indexed corpus.

Focus areas: **prompt caching**, **multi-step reasoning**, **multi-step tool
use**, and how the agent loop is wired — plus the architecture lessons that fall
out of it.

---

## 1. The agent loop

The core loop is the canonical agentic shape (in `harness/agent.py`):

```
while step < MAX_STEPS:
    response = model.create(system, tools, messages)
    append assistant message to `messages`
    if stop_reason != "tool_use":
        break                       # final answer
    for each tool_use block:
        result = dispatch(name, input)
        append tool_result to a user message
    append that user message to `messages`
```

Key mental model: **each step is a fresh, stateless API call.** The model never
"resumes" mid-function. The entire growing `messages` array (plus the cached
stable prefix) *is* the state. The model re-reads the whole conversation every
single step.

That's also why caching matters so much — the large stable manual snapshot is
re-sent on every iteration, so caching it turns N re-sends into 1 write + (N-1)
cheap reads.

### Stop reasons drive the loop

After reading tool results, the assistant generates another message. It is
**one of two things**:

- **More tools** — emits another `tool_use`. `stop_reason == "tool_use"`. Loop
  runs the tool, feeds the result back, calls again.
- **Done** — emits a plain text answer, no tool calls. `stop_reason ==
  "end_turn"`. Loop breaks; that text is the final answer.

So "the assistant response" might be *another tool call*, not necessarily the
final prose. The loop keeps going until a turn has no tool calls.

---

## 2. Tool use round-trip — where tool output goes

This is the part that makes the loop work. After a tool runs:

1. `dispatch()` returns a `ToolResult` (`.text` for the model, `.meta` for the
   UI only — `.meta` never goes to the API).
2. It's wrapped into a `tool_result` block, **keyed by the call's id**:
   ```python
   {
     "type": "tool_result",
     "tool_use_id": block.id,   # matches the tool_use id from the model
     "content": result.text,
     "is_error": result.is_error,
   }
   ```
3. All results from the step are appended to `messages` as a single **`user`**
   message.
4. Loop continues — next `messages.create()` sees the results.

From the API's point of view the conversation looks like:

```
user:      "How do I respond to fault E-12?"
assistant: [thinking][tool_use search_manual id=toolu_01]
user:      [tool_result tool_use_id=toolu_01 "Top results..."]   ← we inject this
assistant: [tool_use get_block id=toolu_02]
user:      [tool_result tool_use_id=toolu_02 "..."]
assistant: "To respond to E-12: ... (cites FAULT-E12, EMRG-03)"  ← end_turn
```

**Tool output is fed back to the model as a `user` message.** "Where does it
go?" → literally back into the conversation history, as if the user said it.

### The `tool_use_id` is the join key

```
assistant:  tool_use   id = "toolu_01ABC"
                          │  (same string)
                          ▼
user:       tool_result tool_use_id = "toolu_01ABC"
```

Anthropic generates the id when it emits the `tool_use`; you echo the identical
string back. Why an explicit id instead of relying on order: **a single
assistant turn can emit multiple tool calls at once (parallel tool use).** With
several results in the next turn, only the ids say which result answers which
call. Position alone isn't safe.

Rules that fall out:
- Every `tool_use` must get a matching `tool_result` in the very next user turn.
  Miss one → API 400s.
- The ids are opaque provider strings — carry them back exactly, don't generate
  or modify them.
- Pairing is by id, not position; but the **assistant-then-user** turn ordering
  is still required, and a `tool_result` must follow the `tool_use` it answers.

---

## 3. Prompt caching

Caching is **prefix caching**: Anthropic caches the request prefix from byte 0
up to and including a `cache_control` marker. A hit requires the prefix to be
**byte-identical** to a prior request.

The layout this demo uses (`harness/context.py`):

```
[ tool definitions ]          stable   ← cache breakpoint
[ system: persona ]           stable
[ system: full manual ]       stable   ← cache breakpoint
------------- cached prefix ends here -------------
[ conversation history ]      volatile (grows each turn)
[ newest user message ]       volatile
```

- **Stable content first, volatile last.** One rotating byte in the prefix
  invalidates the cache for everything after it.
- **Economics:** cache *write* ≈ 1.25× input (one-time), cache *read* ≈ 0.1×
  input (every later turn). Break-even ≈ 1 reuse.
- **Within a turn too:** because each tool-use step is a fresh call with the
  same stable prefix, step 1 writes the cache and steps 2…N read it — caching
  pays *inside* a multi-step turn, not just across turns.
- Toggle caching off in the sidebar and ask the same question to watch the cost
  jump and `cache_read` go to zero.

The token counts shown in the UI come straight from the API `usage` object
(`cache_creation_input_tokens`, `cache_read_input_tokens`). The dollar figures
are illustrative rates (editable in `PRICING`).

---

## 4. The `layers` are UI-only

`ContextBuilder.build()` returns a `BuiltContext` with three fields:

- `system`, `tools` — **actually sent to the API**.
- `layers` — **NOT sent**; a human-readable breakdown of the whole request for
  the Context Inspector tab (name, role, cacheable?, breakpoint?, token est,
  preview). They power the 🟦 cached vs 🟥 volatile coloring and the
  stable/volatile token totals.

Delete the inspector and you could delete `layers` entirely — the agent runs
identically. They're instrumentation, the "made visible" half of the demo. Same
idea as `ToolResult.meta`: machinery vs. instrumentation living side by side.

---

## 5. Tool strategy

Three tools, deliberately shaped to *force* multi-step tool use:

| tool | returns | role |
|---|---|---|
| `search_manual(query, section?)` | ranked block **IDs + snippets** (not full text) | locate |
| `get_block(block_id)` | one block, **full text + its `refs`** | read precisely |
| `get_section(section)` | every block in a section, full | bulk read |

Design choices that matter:
- **Search returns snippets, not full content** — so the agent must take a
  second step (`get_block`) to read the real thing. That's what creates the
  chain. If search returned full text, it'd be one step.
- **`get_block` surfaces `refs`** (`See also: PROP-02, EMRG-03`) — so reading
  one block reveals the next ones to read, producing a 3rd/4th step. Chain depth
  comes from the corpus link graph, not clever tools.
- **Retrieval is plug-and-play** — the seam is `ManualIndex.search() ->
  list[Hit]`. Swap BM25 for cosine / hybrid / a vector DB without touching the
  agent or tools. The query is the contract; the `Hit` shape is the real
  interface (esp. `snippet`, which is load-bearing for the multi-step behavior).
- **Only `search_manual` flows through retrieval.** `get_block` / `get_section`
  read straight from the corpus by ID/section — fetching `FAULT-E12` by id
  shouldn't depend on the ranker. So "swap retrieval" = swap how *search* ranks.

(Currently BM25 only — chosen because retrieval quality isn't what the demo
teaches, and the corpus is keyword-heavy where lexical search shines.)

---

## 6. Provider obligations — opaque state you must round-trip

Some provider data is meaningless to your app but **mandatory to send back**:

- **Thinking signatures** (Anthropic) — opaque `signature` on a thinking block.
  When thinking + tool use are combined, it must be replayed **byte-identical**.
- **Tool-call ids** — provider-generated; the matching result must echo the
  exact id.
- **Ordering constraints** — e.g. the thinking block must come *before* the
  text/tool block in the same assistant turn; preserve order, not just content.
- **`cache_control` placement** — rides on content blocks; some providers /
  gateways only honor it in certain positions.

Rule of thumb: **convert what you understand, preserve what you don't.** The
bugs all live in the state you didn't think you needed to keep.

You can see these live: the **🧾 Raw API response** expander shows the full
`response.model_dump()` per step (and it also prints to the terminal/stderr),
and the **🧠 Reasoning** expander shows each thinking block's `signature`.

---

## 7. Provider-agnostic architecture (anti-corruption layer)

The loop is universal; the wire format is per-provider. So provider types should
live **only at the edge** (invocation / agent-loop level). The moment data
crosses into the app it becomes *your* domain type. Provider types never leak
inward.

```
Anthropic types ─┐
OpenAI types ────┼─► converter (edge only) ─► domain Message ─► rest of app
Gemini types ────┘                              (provider-agnostic, strict)
```

- **Convert both directions.** `toDomain` on the way in, `toProvider` on the way
  out (history has to be replayed back to the provider).
- **Domain type is shaped by use case, not provider** — model only what your app
  does (`TextPart | ToolCall | ToolResult | Reasoning`). Stricter and smaller
  than the provider's superset union; stable when a provider adds block types.
- **One opaque passthrough slot** for provider obligations (§6) — e.g. the
  thinking signature — that you preserve but never interpret. (Vercel calls this
  "provider metadata".)
- Build **converters, not an app built around provider types.** Nothing outside
  the edge should `import anthropic`.

### Same loop, different envelopes

The algorithm ports unchanged across providers. Only serialization differs:

| | Anthropic | OpenAI | Gemini |
|---|---|---|---|
| where tool result goes | `tool_result` block in a **user** msg | dedicated **`role:"tool"`** msg | `functionResponse` part in a user turn |
| id field | `tool_use_id` ↔ `id` | `tool_call_id` ↔ `tool_calls[].id` | historically by **function name**, not id |
| how call is emitted | `tool_use` block in `content` | `tool_calls` array (content often null) | `functionCall` part in `parts` |
| stop reason | `stop_reason: "tool_use"` / `"end_turn"` | `finish_reason: "tool_calls"` / `"stop"` | `finishReason` + functionCall present |
| reasoning round-trip | thinking block + `signature` | o-series reasoning item / `previous_response_id` | "thoughts" |

Libraries like **Vercel AI SDK**, **LiteLLM**, **LangChain** are essentially big
converter layers normalizing exactly these differences. Rolling your own is
reasonable when you want strict, use-case-shaped types instead of a lib's
lowest-common-denominator ones.

---

## 8. Where this harness is intentionally brittle

It's a *teaching* harness — optimized for legibility over defensiveness. Known
gaps (real production work, deliberately omitted so the mechanics stay visible):

- **No retry/backoff** on `messages.create()` — one transient error (rate limit,
  529) kills the turn. Biggest gap.
- **`MAX_STEPS` is a hard wall** — hits the cap and emits a canned string, with
  the last tool results never sent back.
- **`dispatch` itself isn't wrapped** — expected tool errors return
  `is_error=True` (good, model can recover), but a *handler bug* throwing is
  uncaught → turn dies.
- **`max_tokens` cutoff is silent** — `stop_reason == "max_tokens"` is treated
  like a normal finish; truncated answer with no flag.

Things that *look* brittle but are correct:
- **Passing `response.content` back verbatim** — required; thinking blocks must
  be replayed unmodified with tool use.
- **`getattr(block, "type", ...)` sniffing** — works but is the *wrong kind* of
  safe (brittle-permissive). The idiomatic fix is Pydantic + structural pattern
  matching (`match block: case ThinkingBlock(): ...`) and a Pydantic model per
  tool input (`SearchArgs.model_validate(block.input)`) — the Python analog of a
  zod `safeParse` at the boundary. Deferred for now.
```
