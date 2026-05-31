"""A small, readable agent harness for *learning* context engineering.

The package is deliberately split so each concept lives in one file:

- ``corpus``   : the synthetic knowledge base (a starship ops manual)
- ``retrieval``: BM25 search over the manual blocks
- ``tools``    : the 3-tool surface the agent calls + dispatch
- ``context``  : ContextBuilder -- assembles the request and places cache breakpoints
- ``agent``    : the multi-step tool-use loop (real Claude API, extended thinking)

Nothing here is production infrastructure; it is shaped to make the mechanics
of prompt caching, multi-step reasoning, and multi-step tool use *visible*.
"""
