"""The JSON Schema of a result (draft 2020-12). Stable within a schema_version:
fields may be added, none renamed or removed without a new version."""
from __future__ import annotations

import copy
from typing import Any, Dict

_NUM = {"type": ["number", "null"]}
_STR = {"type": ["string", "null"]}
_RANGE = {"type": "array", "items": _NUM, "minItems": 2, "maxItems": 2}
_WHY = {"type": "string", "minLength": 1, "description": "One or two plain sentences: why this matters to the trader."}

VERDICT_CODES = ["real_edge", "cant_tell", "no_edge", "losing", "faded", "too_early"]

_VERDICT = {
    "type": "object",
    "required": ["code", "label", "tone", "headline", "plain", "why"],
    "properties": {
        "code": {"enum": VERDICT_CODES,
                 "description": "real_edge: more than luck. cant_tell: positive but not proven. no_edge: what luck "
                                "does. losing: loses more steadily than luck. faded: worked, then stopped. "
                                "too_early: under 30 trades."},
        "label": {"type": "string"},
        "tone": {"enum": ["good", "warn", "muted", "bad", "faded"]},
        "headline": {"type": "string", "description": "One sentence, safe to quote word for word."},
        "plain": {"type": "string", "description": "2-4 short sentences for a non-coder."},
        "why": _WHY,
        "about": {"type": "string", "description": "With several strategies: the one the code describes."},
    },
}
_STEP = {"type": "object", "required": ["text", "why"],
         "properties": {"text": {"type": "string", "description": "About testing and process; never buy, sell "
                                                                  "or size advice."},
                        "why": _WHY}}
_CHECK = {"type": "object", "required": ["code", "status", "text", "fix", "why"],
          "properties": {"code": {"type": "string"}, "status": {"enum": ["pass", "warn", "fail"]},
                         "text": {"type": "string"}, "fix": _STR, "why": _WHY}}
_QUESTION = {"type": "object", "required": ["id", "field", "question", "why", "answer_with"],
             "properties": {"id": {"type": "string"},
                            "field": {**_STR, "description": "The judge() parameter the answer sets, if any."},
                            "question": {"type": "string"}, "why": _WHY,
                            "answer_with": {"type": "string"}, "impact": {"enum": ["high", "medium", "low"]}}}
_CHANGE = {"type": "object", "required": ["code", "text"],
           "properties": {"code": {"type": "string"}, "text": {"type": "string"}}}
_COMPARE = {
    "type": ["object", "null"],
    "required": ["code", "headline", "why"],
    "properties": {
        "code": {"enum": ["consistent", "worse_than_backtest", "too_early", "units_differ", "need_input"]},
        "headline": {"type": "string"}, "plain": {"type": "string"}, "why": _WHY, "next_step": _STEP,
        "difference": {"type": "object", "properties": {"value": _NUM, "range_95": _RANGE}},
        "odds": _NUM, "e_value_worse": _NUM,
    },
}
_NUMBERS = {
    "type": ["object", "null"],
    "required": ["trades", "unit", "mean_per_trade", "win_rate", "coin_flip", "evidence"],
    "properties": {
        "trades": {"type": "integer"},
        "unit": {"type": "string", "description": "'%' (percent per trade), a currency code, or 'money'."},
        "total": _NUM,
        "mean_per_trade": {"type": "object", "required": ["value", "range_95"],
                           "properties": {"value": _NUM, "range_95": _RANGE}},
        "win_rate": {"type": "object", "properties": {"value": _NUM, "range_95": _RANGE}},
        "coin_flip": {"type": "object", "properties": {"copies": {"type": "integer"},
                                                       "ended_higher": {"type": "integer"},
                                                       "needed_at_most": {"type": "integer"}}},
        "evidence": {"type": "object", "properties": {"e_value": _NUM, "bar": _NUM, "passes": {"type": "boolean"}}},
        "adjusted": {"type": ["object", "null"]},
        "benchmark": {"type": ["object", "null"]},
        "market_adjusted": {"type": ["object", "null"]},
        "trades_needed": {"type": ["integer", "null"]},
    },
}
_CARD = {
    "type": "object",
    "required": ["name", "unit", "n", "fingerprint", "verdict", "next_step", "checks", "ask_the_user",
                 "what_would_change_it", "ai_audit_prompt", "money", "benchmark", "compare", "chart"],
    "properties": {
        "name": {"type": "string"}, "unit": {"type": "string"}, "n": {"type": "integer"},
        "fingerprint": {"type": "string"}, "verdict": _VERDICT,
        "next_step": {"oneOf": [_STEP, {"type": "null"}]},
        "numbers": _NUMBERS,
        "checks": {"type": "array", "items": _CHECK},
        "ask_the_user": {"type": "array", "items": _QUESTION},
        "what_would_change_it": {"type": "array", "items": _CHANGE},
        "ai_audit_prompt": _STR,
        "money": {"type": ["object", "null"]},
        "benchmark": {"type": ["object", "null"]},
        "compare": _COMPARE,
        "chart": {"type": ["object", "null"],
                  "properties": {"you": {"type": "array"}, "fan": {"type": "array"}}},
    },
}

SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://github.com/Sakke-code/edge-or-luck/schema/result-1.0.json",
    "title": "edge or luck: result",
    "description": "Whether a trading record is a real edge or luck, in fields an assistant can relay without "
                   "computing anything else. Start with verdict.headline, then next_step and ask_the_user.",
    "type": "object",
    "required": ["schema_version", "engine_version", "status", "input_fingerprint", "verdict", "next_step",
                 "numbers", "checks", "ask_the_user", "what_would_change_it", "ai_audit_prompt",
                 "for_the_assistant", "reply_draft", "summary", "strategies", "portfolio", "compare", "input",
                 "settings", "warnings", "need"],
    "properties": {
        "schema_version": {"const": "1.0"},
        "engine_version": {"type": "string"},
        "status": {"enum": ["ok", "need_input", "error"]},
        "input_fingerprint": {**_STR, "description": "sha256 of the trades as read; the same input gives the "
                                                     "same result."},
        "verdict": {"oneOf": [_VERDICT, {"type": "null"}]},
        "next_step": {"oneOf": [_STEP, {"type": "null"}]},
        "numbers": {**_NUMBERS, "description": "The single strategy's numbers; null when there are several "
                                               "(see strategies[].numbers)."},
        "checks": {"type": "array", "items": _CHECK},
        "ask_the_user": {"type": "array", "items": _QUESTION},
        "what_would_change_it": {"type": "array", "items": _CHANGE},
        "ai_audit_prompt": {**_STR, "description": "A copy-paste prompt for auditing the bot's code, when the "
                                                   "record looks too good or a backtest passes."},
        "for_the_assistant": {"type": "array", "items": {"type": "string"}},
        "reply_draft": {"type": "string", "description": "A short answer the assistant may adapt."},
        "how_we_know": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "summary": {"type": "object", "required": ["headline", "text", "counts"]},
        "strategies": {"type": "array", "items": _CARD},
        "portfolio": {"type": "object", "properties": {"pairs": {"type": "array"}}},
        "compare": _COMPARE,
        "input": {"type": "object"},
        "settings": {"type": "object"},
        "warnings": {"type": "array", "items": {"type": "string"}},
        "need": {"type": ["object", "null"],
                 "properties": {"problem": {"type": "string"}, "needs": {"type": "string"},
                                "columns_seen": {"type": "array"}, "question_for_trader": {"type": "string"}}},
    },
}


def schema() -> Dict[str, Any]:
    """The JSON Schema every result follows."""
    return copy.deepcopy(SCHEMA)
