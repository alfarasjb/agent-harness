"""Synthetic knowledge base: the *ACV Helios-IX Operations Manual*.

This is invented from scratch -- a deep-space survey vessel's ops manual. It is
built to exercise the three things the demo teaches:

- **Multi-step tool use**: fault entries reference subsystem and emergency blocks
  by ID, so answering "how do I handle fault E-12?" naturally requires several
  tool calls (search -> read fault -> read the procedures it points to).
- **Multi-step reasoning**: questions like "can I jump while the coolant loop is
  degraded?" force the model to combine constraints from several blocks.
- **Prompt caching**: the whole manual is a stable prefix (~several thousand
  tokens) re-sent every turn, which is exactly what caching is for.

Each block has a stable ID (e.g. ``PROP-02``, ``FAULT-E12``) so the agent can
fetch precise context, and ``refs`` to model cross-document links.
"""

from __future__ import annotations

from dataclasses import dataclass ,field


SECTIONS =[
"OVERVIEW",
"PROPULSION",
"POWER",
"LIFE_SUPPORT",
"NAVIGATION",
"FAULT_CODES",
"EMERGENCY_PROCEDURES",
"CREW",
"GLOSSARY",
]


@dataclass (frozen =True )
class Block :
    """One addressable unit of the manual."""

    id :str
    section :str
    title :str
    text :str
    refs :tuple [str ,...]=field (default_factory =tuple )

    def render (self )->str :
        """Full rendering used by get_block / the context snapshot."""
        ref_line =f"\nSee also: {', '.join (self .refs )}"if self .refs else ""
        return f"[{self .id } | {self .section }] {self .title }\n{self .text }{ref_line }"


_BLOCKS :list [Block ]=[

Block (
"OVR-01",
"OVERVIEW",
"Vessel summary",
"The ACV Helios-IX is a Class-IV deep-space survey vessel with a crew "
"complement of nine. Primary systems are the Mark-3 flux drive, a twin "
"fusion reactor pair (REAC-A and REAC-B), a closed-loop life-support "
"stack, and the Astra navigation suite. The ship operates autonomously "
"for missions up to 400 standard days.",
),
Block (
"OVR-02",
"OVERVIEW",
"Operating modes",
"The Helios-IX has four operating modes: STATION-KEEPING (idle, minimal "
"draw), CRUISE (sub-light maneuvering), SURVEY (sensors prioritized, "
"drive locked out), and JUMP (flux drive engaged). Mode transitions are "
"gated by interlocks documented per subsystem. JUMP requires a healthy "
"flux margin and both coolant loops nominal.",
refs =("PROP-01","PROP-02","GLO-flux-margin"),
),

Block (
"PROP-01",
"PROPULSION",
"Mark-3 flux drive overview",
"The Mark-3 flux drive folds local space to achieve faster-than-light "
"transit. It draws from both reactors during spin-up and requires the "
"flux margin to stay above 0.15 throughout the jump. The drive is "
"locked out in SURVEY mode and during any active red-tier fault.",
refs =("GLO-flux-margin","OVR-02","FAULT-E12"),
),
Block (
"PROP-02",
"PROPULSION",
"Coolant loops A, B, and C",
"Three independent coolant loops service the drive. Loops A and B cool "
"the field coils; loop C cools the injector array. JUMP mode requires "
"loops A and B nominal. Loop C may be degraded for a jump but limits "
"sustained drive time to 20 minutes. A coolant loop is 'degraded' when "
"flow drops below 60% of rated.",
refs =("FAULT-E12","FAULT-E07","EMRG-03"),
),
Block (
"PROP-03",
"PROPULSION",
"Drive spin-up sequence",
"Spin-up: (1) confirm both reactors above 80% output, (2) arm the field "
"coils, (3) verify flux margin > 0.20 pre-jump, (4) release the JUMP "
"interlock. Abort spin-up immediately if the flux margin dips below "
"0.15 or any coolant loop trips.",
refs =("POW-01","GLO-flux-margin","PROP-02"),
),

Block (
"POW-01",
"POWER",
"Twin fusion reactors",
"REAC-A and REAC-B each supply up to 1.2 GW. Either reactor alone can "
"sustain CRUISE and SURVEY modes, but JUMP requires both online above "
"80%. Reactors share a common deuterium feed with isolation valves "
"IV-A and IV-B.",
refs =("FAULT-E04","PROP-03"),
),
Block (
"POW-02",
"POWER",
"Power priority ladder",
"On a power deficit, the bus sheds load in this order: (1) science "
"sensors, (2) non-essential lighting, (3) artificial gravity, (4) drive "
"pre-charge. Life support and command systems are never shed "
"automatically.",
refs =("LIFE-01","FAULT-E04"),
),

Block (
"LIFE-01",
"LIFE_SUPPORT",
"Atmosphere processing",
"The life-support stack scrubs CO2 and regenerates O2 via the dual "
"Sabatier units S-1 and S-2. A single unit sustains the full crew "
"indefinitely; both running provides margin. Cabin O2 target is 21%.",
refs =("FAULT-E18","EMRG-01"),
),
Block (
"LIFE-02",
"LIFE_SUPPORT",
"Water and thermal",
"Potable water is recovered at 94% efficiency. Crew cabin thermal load "
"is dumped through radiator panels R-1..R-4. Loss of two or more "
"radiators requires load reduction within 30 minutes.",
refs =("FAULT-E18",),
),

Block (
"NAV-01",
"NAVIGATION",
"Astra navigation suite",
"The Astra suite fuses star-tracker, inertial, and pulsar-timing inputs "
"to fix position. A valid fix requires at least two of the three inputs "
"agreeing within tolerance. JUMP is interlocked against a stale fix "
"(older than 60 seconds).",
refs =("FAULT-E09","OVR-02"),
),
Block (
"NAV-02",
"NAVIGATION",
"Jump plotting",
"A jump plot is valid for 60 seconds. Replot if the vessel maneuvers or "
"the fix goes stale. The plotter refuses destinations inside a stellar "
"gravity well above threshold G-2.",
refs =("NAV-01","PROP-03"),
),

Block (
"FAULT-E04",
"FAULT_CODES",
"Fault E-04: reactor output sag",
"Cause: one reactor drops below 80% under load. Tier: amber. Effect: "
"JUMP interlock engages until both reactors recover. Action: check the "
"deuterium feed and isolation valves; if a reactor is offline, follow "
"the reactor restart procedure.",
refs =("POW-01","EMRG-02"),
),
Block (
"FAULT-E07",
"FAULT_CODES",
"Fault E-07: coolant flow low",
"Cause: a coolant loop drops below 60% rated flow. Tier: amber. Effect: "
"affected loop flagged degraded. If loop A or B, JUMP is locked out. "
"Action: inspect pump and lines; purge if cavitation suspected.",
refs =("PROP-02","EMRG-03"),
),
Block (
"FAULT-E09",
"FAULT_CODES",
"Fault E-09: navigation fix stale",
"Cause: fewer than two nav inputs agree, or the fix is older than 60s. "
"Tier: amber. Effect: JUMP interlocked. Action: re-acquire star-tracker "
"lock; if pulsar timing is the disagreeing input, deweight it and "
"replot.",
refs =("NAV-01","NAV-02"),
),
Block (
"FAULT-E12",
"FAULT_CODES",
"Fault E-12: flux margin critical",
"Cause: flux margin falls below 0.15 during or before a jump. Tier: "
"RED. Effect: drive lockout and automatic spin-down. Action: abort the "
"jump immediately, restore coolant flow on loops A and B, and follow "
"the emergency drive-abort procedure before any retry.",
refs =("PROP-01","PROP-02","EMRG-03"),
),
Block (
"FAULT-E18",
"FAULT_CODES",
"Fault E-18: atmosphere out of band",
"Cause: cabin O2 outside 19.5%-23.5% or CO2 above limit. Tier: RED. "
"Effect: alarm and forced ventilation. Action: confirm both Sabatier "
"units; if a unit is down, follow the life-support contingency "
"procedure and don respirators if O2 keeps falling.",
refs =("LIFE-01","EMRG-01"),
),

Block (
"EMRG-01",
"EMERGENCY_PROCEDURES",
"Life-support contingency",
"If one Sabatier unit fails: (1) bring the standby unit to full output, "
"(2) reduce crew activity to lower CO2 generation, (3) if cabin O2 "
"trends below 19.5%, distribute respirators and bleed the O2 reserve "
"tank. Do not vent cabin atmosphere except on a fire emergency.",
refs =("LIFE-01","FAULT-E18"),
),
Block (
"EMRG-02",
"EMERGENCY_PROCEDURES",
"Reactor restart",
"To restart an offline reactor: (1) confirm isolation valve open, (2) "
"verify deuterium feed pressure, (3) initiate ignition sequence, (4) "
"ramp to 80% before re-arming the JUMP interlock. Never restart with a "
"suspected feed-line breach.",
refs =("POW-01","FAULT-E04"),
),
Block (
"EMRG-03",
"EMERGENCY_PROCEDURES",
"Emergency drive abort",
"On a red-tier drive fault: (1) command spin-down, (2) safe the field "
"coils, (3) restore coolant flow on loops A and B to nominal, (4) clear "
"the fault and re-verify flux margin > 0.20 before any retry. A retry "
"with margin between 0.15 and 0.20 is prohibited.",
refs =("PROP-01","PROP-02","FAULT-E12"),
),

Block (
"CREW-01",
"CREW",
"Watch standing",
"The ship runs three watches of three crew. The command watch officer "
"authorizes mode transitions. JUMP additionally requires the engineer "
"of the watch to confirm drive and coolant status.",
refs =("OVR-02",),
),

Block (
"GLO-flux-margin",
"GLOSSARY",
"Term: flux margin",
"Flux margin is the dimensionless ratio of available field-coil headroom "
"to the demand of the current jump plot. Below 0.15 the drive cannot "
"safely hold the fold and trips fault E-12. A pre-jump check requires "
"> 0.20.",
refs =("PROP-01","FAULT-E12"),
),
Block (
"GLO-degraded",
"GLOSSARY",
"Term: degraded (coolant loop)",
"A coolant loop is 'degraded' when measured flow is below 60% of rated. "
"A degraded loop A or B locks out JUMP; a degraded loop C only limits "
"sustained drive time.",
refs =("PROP-02","FAULT-E07"),
),
]


BLOCKS :dict [str ,Block ]={b .id :b for b in _BLOCKS }


def blocks_in_section (section :str )->list [Block ]:
    return [b for b in _BLOCKS if b .section ==section ]


def all_blocks ()->list [Block ]:
    return list (_BLOCKS )


def render_full_manual ()->str :
    """The stable snapshot injected into the context every turn (and cached)."""
    parts =["ACV HELIOS-IX OPERATIONS MANUAL","="*40 ,""]
    for section in SECTIONS :
        parts .append (f"## {section }")
        for b in blocks_in_section (section ):
            parts .append (b .render ())
            parts .append ("")
    return "\n".join (parts )
