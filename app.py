"""Streamlit demo: a window into context engineering.

Run:  streamlit run app.py   (needs ANTHROPIC_API_KEY)

Three things this app makes visible:
  1. Prompt caching      -- real cache_read / cache_write tokens, hit rate, and
                            cost with vs without caching (toggle it and compare).
  2. Multi-step reasoning -- the model's extended thinking, per step.
  3. Multi-step tool use  -- the search -> read -> read chain over the manual.

Plus a Context Inspector showing exactly what is sent each turn, split into the
cached stable prefix and the volatile tail.
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from anthropic import Anthropic

from harness .agent import Agent ,AgentConfig ,MODELS ,Usage
from harness import corpus





def _load_dotenv ()->None :
    """Tiny .env loader so a key in ./.env works without extra deps."""
    env_path =Path (__file__ ).parent /".env"
    if "ANTHROPIC_API_KEY"in os .environ or not env_path .exists ():
        return
    for line in env_path .read_text (encoding ="utf-8").splitlines ():
        line =line .strip ()
        if not line or line .startswith ("#")or "="not in line :
            continue
        key ,_ ,value =line .partition ("=")
        os .environ .setdefault (key .strip (),value .strip ())


_load_dotenv ()

st .set_page_config (page_title ="Context Engineering Demo",page_icon ="🛰️",layout ="wide")

SAMPLE_QUESTIONS =[
"How do I respond to fault E-12?",
"Can I jump while coolant loop C is degraded?",
"Where is 'flux margin' defined, and what's the safe pre-jump value?",
"What's in the life support section?",
"A reactor dropped to 70%. Can I still jump, and what should I do?",
]





def _init_state ()->None :
    ss =st .session_state
    ss .setdefault ("history",[])
    ss .setdefault ("transcript",[])
    ss .setdefault ("last_context",None )
    ss .setdefault ("session_usage",Usage ())
    ss .setdefault ("session_model",MODELS [0 ])
    ss .setdefault ("queued_question",None )


_init_state ()


def reset_conversation ()->None :
    st .session_state .history =[]
    st .session_state .transcript =[]
    st .session_state .last_context =None
    st .session_state .session_usage =Usage ()





with st .sidebar :
    st .header ("🛰️ Agent harness")
    st .caption ("ACV Helios-IX ops assistant")

    has_key =bool (os .environ .get ("ANTHROPIC_API_KEY"))
    if has_key :
        st .success ("ANTHROPIC_API_KEY detected",icon ="🔑")
    else :
        st .error ("Set ANTHROPIC_API_KEY to run (this demo is live-only).",icon ="🚫")

    st .divider ()
    model =st .selectbox ("Model",MODELS ,index =0 )
    cache_enabled =st .toggle (
    "Prompt caching",value =True ,
    help ="Place cache_control breakpoints on the stable prefix (tools + "
    "system + manual). Turn off to see the cost without caching.",
    )
    thinking_enabled =st .toggle (
    "Extended thinking",value =True ,
    help ="Enable multi-step reasoning. The model's thinking is shown per step.",
    )
    thinking_budget =st .slider (
    "Thinking budget (tokens)",1024 ,6000 ,1500 ,step =256 ,
    disabled =not thinking_enabled ,
    )

    st .divider ()
    st .caption ("Try a question:")
    for q in SAMPLE_QUESTIONS :
        if st .button (q ,use_container_width =True ,key =f"sample::{q }"):
            st .session_state .queued_question =q
            st .rerun ()

    st .divider ()
    if st .button ("🔄 Reset conversation",use_container_width =True ):
        reset_conversation ()
        st .rerun ()


    if model !=st .session_state .session_model :
        st .session_state .session_model =model
        reset_conversation ()


config =AgentConfig (
model =model ,
cache_enabled =cache_enabled ,
thinking_enabled =thinking_enabled ,
thinking_budget =thinking_budget ,
)





def _render_search_hits (meta :dict )->None :
    hits =meta .get ("hits")
    if not hits :
        return
    st .table (
    [
    {
    "rank":h ["rank"],
    "block":h ["block_id"],
    "section":h ["section"],
    "score":h ["score"],
    "title":h ["title"],
    }
    for h in hits
    ]
    )


def _render_step (step :dict )->None :
    n =step ["n"]
    for think in step ["thinking"]:
        with st .expander (f"🧠 Reasoning — step {n }",expanded =False ):
            st .markdown (think )
    for txt in step ["texts"]:
        if txt .strip ():
            st .markdown (txt )
    for call in step ["tools"]:
        icon ="⚠️"if call ["is_error"]else "🔧"
        st .markdown (f"{icon } **`{call ['name']}`** `{call ['input']}`")
        label ="tool error"if call ["is_error"]else "tool result"
        with st .expander (f"↳ {label }: {call ['name']} (step {n })",expanded =False ):
            if call ["meta"].get ("hits")is not None :
                _render_search_hits (call ["meta"])
            st .code (call ["text"],language ="text")


def _render_metrics (usage :Usage ,steps :int ,model :str ,cached :bool )->None :
    costs =usage .costs (model )
    cols =st .columns (6 )
    cols [0 ].metric ("Steps",steps )
    cols [1 ].metric ("Cache write",f"{usage .cache_creation_input_tokens :,}")
    cols [2 ].metric ("Cache read",f"{usage .cache_read_input_tokens :,}")
    cols [3 ].metric ("Hit rate",f"{usage .cache_hit_rate :.0%}")
    cols [4 ].metric ("Cost",f"${costs ['actual']:.5f}")
    delta =f"-${costs ['saved']:.5f}"if cached else None
    cols [5 ].metric ("vs uncached",f"${costs ['uncached']:.5f}",delta =delta )


def _render_assistant_record (record :dict )->None :
    for step in record ["steps"]:
        _render_step (step )
    st .markdown ("---")
    st .markdown (record ["answer"]or "_(no answer)_")
    _render_metrics (
    record ["usage"],record ["steps_count"],record ["model"],record ["cached"]
    )


def render_transcript ()->None :
    for turn in st .session_state .transcript :
        if turn ["role"]=="user":
            with st .chat_message ("user"):
                st .markdown (turn ["text"])
        else :
            with st .chat_message ("assistant"):
                _render_assistant_record (turn ["record"])





def execute_turn (user_text :str )->None :
    client =Anthropic ()
    agent =Agent (client ,config )

    steps_by_n :dict [int ,dict ]={}
    pending_inputs :dict [str ,dict ]={}
    record ={
    "steps":[],
    "answer":"",
    "usage":Usage (),
    "steps_count":0 ,
    "model":config .model ,
    "cached":config .cache_enabled ,
    }

    with st .chat_message ("assistant"):
        with st .status ("Running the agent loop…",expanded =True )as status :
            try :
                for event in agent .run_turn (st .session_state .history ,user_text ):
                    etype =event ["type"]
                    if etype =="context":
                        st .session_state .last_context =event ["built"]
                    elif etype =="step":
                        n =event ["n"]
                        steps_by_n [n ]={"n":n ,"thinking":[],"texts":[],"tools":[]}
                        status .update (label =f"Step {n }: calling the model…")
                    elif etype =="thinking":
                        steps_by_n [event ["step"]]["thinking"].append (event ["text"])
                        status .write (f"🧠 step {event ['step']}: reasoning…")
                    elif etype =="text":
                        steps_by_n [event ["step"]]["texts"].append (event ["text"])
                    elif etype =="tool_call":
                        pending_inputs [event ["id"]]=event ["input"]
                        status .write (f"🔧 {event ['name']}({event ['input']})")
                    elif etype =="tool_result":
                        steps_by_n [event ["step"]]["tools"].append (
                        {
                        "name":event ["name"],
                        "input":pending_inputs .get (event ["id"],{}),
                        "text":event ["text"],
                        "meta":event ["meta"],
                        "is_error":event ["is_error"],
                        }
                        )
                    elif etype =="usage":
                        pass
                    elif etype =="done":
                        result =event ["result"]
                        record ["answer"]=result .answer
                        record ["usage"]=result .usage
                        record ["steps_count"]=result .steps
                        st .session_state .history =result .messages
                        st .session_state .session_usage .add (result .usage )
                status .update (label ="Done",state ="complete",expanded =False )
            except Exception as exc :
                status .update (label ="Error",state ="error")
                st .error (f"{type (exc ).__name__ }: {exc }")
                record ["answer"]=f"⚠️ Request failed: {exc }"

    record ["steps"]=[steps_by_n [k ]for k in sorted (steps_by_n )]

    st .session_state .transcript .append ({"role":"user","text":user_text })
    st .session_state .transcript .append ({"role":"assistant","record":record })





st .title ("Context Engineering, made visible")
st .caption (
"A real Claude agent answering questions about a synthetic starship manual — "
"instrumented so you can watch prompt caching, multi-step reasoning, and "
"multi-step tool use as they happen."
)

tab_chat ,tab_inspector ,tab_about =st .tabs (
["💬 Chat","🔍 Context inspector","📖 How it works"]
)

with tab_chat :
    render_transcript ()


    su =st .session_state .session_usage
    if su .input_tokens or su .cache_read_input_tokens or su .cache_creation_input_tokens :
        costs =su .costs (config .model )
        st .divider ()
        c =st .columns (4 )
        c [0 ].metric ("Session cache reads",f"{su .cache_read_input_tokens :,}")
        c [1 ].metric ("Session hit rate",f"{su .cache_hit_rate :.0%}")
        c [2 ].metric ("Session cost",f"${costs ['actual']:.5f}")
        c [3 ].metric ("Saved by caching",f"${costs ['saved']:.5f}")

with tab_inspector :
    built =st .session_state .last_context
    st .subheader ("What gets sent to the model each turn")
    if built is None :
        st .info ("Ask a question first — the assembled context will appear here.")
    else :
        st .markdown (
        f"**Stable prefix:** ~{built .stable_tokens :,} tok "
        f"· **Volatile tail:** ~{built .volatile_tokens :,} tok  \n"
        "The stable prefix is byte-identical turn to turn, so with caching on "
        "it is written once and *read* (at 10% cost) on every later turn."
        )
        for layer in built .layers :
            if layer .role in ("tools","system"):
                tone ="🟦 cached"if layer .cacheable else "⬜ not cached"
            else :
                tone ="🟥 volatile"
            bp ="  ⟵ **cache breakpoint**"if layer .cache_breakpoint else ""
            with st .expander (
            f"{tone } · {layer .name } · ~{layer .tokens :,} tok{bp }",expanded =False
            ):
                st .code (layer .preview ,language ="text")
        st .caption (
        "Order matters: stable content first, volatile last. A single "
        "rotating byte in the prefix invalidates the cache for everything "
        "after it."
        )

with tab_about :
    st .markdown (
    """
### The three mechanics on display

**1 · Prompt caching.** The tool definitions, system persona, and the full
manual snapshot form a *stable prefix*. With caching on, `ContextBuilder`
attaches `cache_control: {type: "ephemeral"}` markers so Anthropic caches that
prefix. Turn 1 *writes* the cache (you'll see `cache_creation` tokens at 1.25×
cost); every turn after *reads* it (`cache_read` tokens at 0.1× cost). Flip the
**Prompt caching** toggle and ask the same question twice to watch the cost line
move. The conversation tail and newest user message sit *after* the last
breakpoint, so they're never cached — correct, since they change every turn.

**2 · Multi-step reasoning.** With **Extended thinking** on, the model reasons
before acting. Each step's thinking is shown in a 🧠 expander. Watch it plan
which tool to call and how to combine constraints from several blocks.

**3 · Multi-step tool use.** The agent has three tools — `search_manual`,
`get_section`, `get_block`. A question like *"how do I respond to fault E-12?"*
makes it search, read the fault block, then follow that block's cross-references
to the procedures it points to — several tool rounds before answering.

### The knowledge base
A synthetic ops manual for the *ACV Helios-IX* — propulsion, power, life
support, navigation, fault codes, emergency procedures, glossary. Blocks
cross-reference each other by ID, which is what makes the tool chains deep.
        """
    )
    st .caption (
    f"Manual: {len (corpus .all_blocks ())} blocks across "
    f"{len (corpus .SECTIONS )} sections."
    )




queued =st .session_state .queued_question
st .session_state .queued_question =None
typed =st .chat_input ("Ask the Helios-IX ops assistant…",disabled =not has_key )
user_text =typed or queued

if user_text and has_key :
    execute_turn (user_text )
    st .rerun ()
