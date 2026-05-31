# Context Engineering, made visible 🛰️

A small Streamlit app that turns an agent harness inside-out so you can *watch*
the three mechanics that matter for context engineering:

1. **Prompt caching** — real `cache_read` / `cache_write` token counts, hit rate,
   and cost **with vs without** caching (toggle it and compare on the same
   question).
2. **Multi-step reasoning** — the model's extended thinking, shown per step.
3. **Multi-step tool use** — the `search → read → read` chain over a corpus.

The agent answers questions about a **synthetic starship operations manual**
(the *ACV Helios-IX*). The niche is just a vehicle — the manual's blocks
cross-reference each other by ID, which is what makes the tool chains deep and
the cached prefix large.

> This demo makes **real Claude API calls** — prompt caching is only truly
> demonstrable live, because the cache token counts come back from the API.

---

## Setup

```bash
pip install -r requirements.txt

# provide your key (either works):
#   - copy .env.example to .env and fill it in, or
#   - set the env var directly
$env:ANTHROPIC_API_KEY = "sk-ant-..."   # PowerShell
export ANTHROPIC_API_KEY="sk-ant-..."    # bash

streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501).

---

## How to drive the demo

Start with a sample question from the sidebar, e.g. **"How do I respond to
fault E-12?"** Then:

- **Watch the tool chain.** The agent searches the manual, reads `FAULT-E12`,
  and follows its `See also` cross-references (`EMRG-03`, `PROP-02`) before
  answering — several tool rounds, each in the trace.
- **Watch the reasoning.** Open the 🧠 expanders to see the model plan each step
  (with *Extended thinking* on).
- **See caching pay off.** Ask a **second** question. Turn 1 shows
  `cache write` tokens (the manual gets written to cache at 1.25×); turn 2 shows
  `cache read` tokens (read back at 0.1×) and a high hit rate. The **vs uncached**
  metric shows what you'd have paid without caching.
- **Prove it.** Toggle **Prompt caching off** in the sidebar and ask again —
  watch the cost jump and `cache read` go to zero.
- **Open the 🔍 Context inspector** to see exactly what's sent each turn, split
  into the 🟦 cached stable prefix (tools + system + manual) and the 🟥 volatile
  tail (history + newest message), with the cache breakpoints marked.

---

## How it's built

```
app.py                 Streamlit UI: chat, live trace, cache meter, inspector
harness/
  corpus.py            the synthetic Helios-IX manual (blocks + cross-refs)
  retrieval.py         BM25 index over the blocks
  tools.py             3-tool surface (search_manual / get_section / get_block)
  context.py           ContextBuilder — assembles request, places cache breakpoints
  agent.py             multi-step tool-use loop (real API, extended thinking)
```

### The cache layout (`context.py`)

```
[ tool definitions ]          🟦 stable   ← cache breakpoint #1
[ system: persona ]           🟦 stable
[ system: full manual ]       🟦 stable   ← cache breakpoint #2
------------- cached prefix ends here -------------
[ conversation history ]      🟥 volatile (grows each turn)
[ newest user message ]       🟥 volatile
```

Anthropic caches the request **prefix** up to each `cache_control` marker. A hit
requires that prefix to be **byte-identical** to a previous request — so stable
content goes first and volatile content last. One rotating byte in the prefix
invalidates everything after it.

### Notes / knobs

- **Model:** Sonnet 4.6 (default), Opus 4.8, or Haiku 4.5. Switching models
  resets the conversation (different models can't share a cache prefix).
- **Pricing** in `harness/agent.py` (`PRICING`) is illustrative 2026 rates,
  editable — the *mechanics* are the point, not the exact dollars. Cache write =
  1.25× input, cache read = 0.1× input.
- **Cost figures are estimates** derived from real token counts × those rates.
  The token counts themselves come straight from the API `usage` object.
