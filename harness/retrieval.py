"""BM25 retrieval over the manual blocks.

Kept deliberately simple: a lexical index the agent drives by emitting query
terms. The point of the demo is the *agent loop* over retrieval, not the
retrieval algorithm itself -- so this is one clear, inspectable ranker rather
than a hybrid stack.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .import corpus

_TOKEN_RE =re .compile (r"[a-z0-9]+")


def _tokenize (text :str )->list [str ]:
    return _TOKEN_RE .findall (text .lower ())


@dataclass
class Hit :
    block_id :str
    title :str
    section :str
    snippet :str
    score :float
    rank :int


class ManualIndex :
    """In-process BM25 index built once over all blocks."""

    def __init__ (self )->None :
        self ._blocks =corpus .all_blocks ()

        self ._docs =[
        _tokenize (f"{b .id } {b .title } {b .text }")for b in self ._blocks
        ]
        self ._bm25 =BM25Okapi (self ._docs )

    def search (
    self ,query :str ,section :str |None =None ,k :int =5
    )->list [Hit ]:
        scores =self ._bm25 .get_scores (_tokenize (query ))
        order =sorted (range (len (self ._blocks )),key =lambda i :scores [i ],reverse =True )

        hits :list [Hit ]=[]
        for i in order :
            block =self ._blocks [i ]
            if section and block .section !=section :
                continue
            if scores [i ]<=0 :
                continue
            hits .append (
            Hit (
            block_id =block .id ,
            title =block .title ,
            section =block .section ,
            snippet =_snippet (block .text ),
            score =round (float (scores [i ]),3 ),
            rank =len (hits )+1 ,
            )
            )
            if len (hits )>=k :
                break
        return hits


def _snippet (text :str ,limit :int =160 )->str :
    text =" ".join (text .split ())
    return text if len (text )<=limit else text [:limit ].rstrip ()+"..."



INDEX =ManualIndex ()
