"""The agent's tool surface: three tools over the manual.

This mirrors the classic search/read split that makes multi-step tool use
natural:

- ``search_manual``      -> find relevant blocks (ranked IDs + snippets)
- ``get_section``        -> bulk pull a whole section
- ``get_block``          -> fetch one block in full by ID

The agent typically chains them: search to locate, then get_block to read the
exact cross-referenced procedures. Each dispatch returns both a string (fed
back to the model) and structured ``meta`` (rendered in the trace UI).
"""

from __future__ import annotations

from typing import Any

from .import corpus
from .retrieval import INDEX



TOOL_SCHEMAS :list [dict [str ,Any ]]=[
{
"name":"search_manual",
"description":(
"Search the Helios-IX operations manual for blocks relevant to a "
"query. Returns ranked block IDs with short snippets. Use this "
"first to locate where something is documented, then get_block to "
"read the full text. Optionally restrict to one section."
),
"input_schema":{
"type":"object",
"properties":{
"query":{
"type":"string",
"description":"Keywords describing what to find.",
},
"section":{
"type":"string",
"enum":corpus .SECTIONS ,
"description":"Optional section filter.",
},
},
"required":["query"],
},
},
{
"name":"get_section",
"description":(
"Return every block in a section, in full. Use when the user asks "
"about a whole section (e.g. 'what's in the propulsion section?')."
),
"input_schema":{
"type":"object",
"properties":{
"section":{"type":"string","enum":corpus .SECTIONS },
},
"required":["section"],
},
},
{
"name":"get_block",
"description":(
"Fetch one manual block in full by its ID (e.g. 'FAULT-E12'). Use "
"this to read a specific block, including ones cross-referenced by "
"another block you already read."
),
"input_schema":{
"type":"object",
"properties":{
"block_id":{"type":"string","description":"e.g. PROP-02"},
},
"required":["block_id"],
},
},
]





class ToolResult :
    """What a tool call produces: text for the model + meta for the UI."""

    def __init__ (self ,text :str ,meta :dict [str ,Any ],is_error :bool =False ):
        self .text =text
        self .meta =meta
        self .is_error =is_error


def dispatch (name :str ,tool_input :dict [str ,Any ])->ToolResult :
    if name =="search_manual":
        return _search (tool_input )
    if name =="get_section":
        return _get_section (tool_input )
    if name =="get_block":
        return _get_block (tool_input )
    return ToolResult (
    f"Unknown tool: {name }",{"error":f"unknown tool {name }"},is_error =True
    )


def _search (args :dict [str ,Any ])->ToolResult :
    query =(args .get ("query")or "").strip ()
    section =args .get ("section")
    if not query :
        return ToolResult ("Error: empty query.",{"error":"empty query"},True )

    hits =INDEX .search (query ,section =section ,k =5 )
    if not hits :
        return ToolResult (
        "No matching blocks found.",
        {"query":query ,"section":section ,"hits":[]},
        )

    lines =[f"Top results for '{query }'"+(f" in {section }"if section else "")]
    for h in hits :
        lines .append (f"  {h .rank }. {h .block_id } ({h .section }) -- {h .title }")
        lines .append (f"     {h .snippet }")
    meta ={
    "query":query ,
    "section":section ,
    "hits":[
    {
    "rank":h .rank ,
    "block_id":h .block_id ,
    "section":h .section ,
    "title":h .title ,
    "score":h .score ,
    "snippet":h .snippet ,
    }
    for h in hits
    ],
    }
    return ToolResult ("\n".join (lines ),meta )


def _get_section (args :dict [str ,Any ])->ToolResult :
    section =args .get ("section")
    if section not in corpus .SECTIONS :
        return ToolResult (
        f"Error: unknown section '{section }'.",
        {"error":f"unknown section {section }"},
        True ,
        )
    blocks =corpus .blocks_in_section (section )
    text ="\n\n".join (b .render ()for b in blocks )
    return ToolResult (
    text ,
    {"section":section ,"block_ids":[b .id for b in blocks ]},
    )


def _get_block (args :dict [str ,Any ])->ToolResult :
    block_id =(args .get ("block_id")or "").strip ()
    block =corpus .BLOCKS .get (block_id )
    if not block :
        return ToolResult (
        f"Error: no block with ID '{block_id }'.",
        {"error":f"unknown block {block_id }","block_id":block_id },
        True ,
        )
    return ToolResult (
    block .render (),
    {"block_id":block .id ,"section":block .section ,"refs":list (block .refs )},
    )
