"""What a planned run would cost in model calls, with the arithmetic shown.

Used by the dry runs of legend_evergreen.py and legend_child_bio.py. Input
tokens come from the prompts the run would actually send (their characters,
at CHARS_PER_TOKEN). Output cannot be known before the call, so every step
gives two figures: a low one (the visible answer it is expected to return,
no thinking) and a high one (the step's output cap, which the call cannot
exceed). On Gemini 2.5 the model's thinking is billed as output and counts
against the cap, so the true cost lies between the two.

Prices are list prices for standard (online) requests, USD per million
tokens, as checked on PRICES_CHECKED_ON at PRICES_SOURCE. The Vertex AI
pricing page did not render to the checker that day; the Gemini API's page
gives the same list prices Vertex has charged for these models. That page
also marks gemini-2.5-pro and gemini-2.5-flash deprecated, with requests to
route to newer models from 2026-12-01: a run after that date should be
priced again. A model missing here is not guessed: pass its prices with
--price MODEL=INPUT/OUTPUT.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

PRICES_CHECKED_ON = "2026-11-12"
PRICES_SOURCE = "https://aistudio.google.com/docs/pricing (paid tier, standard)"

# model -> (input, output) USD per 1M tokens; output includes thinking.
PRICES: Dict[str, Tuple[float, float]] = {
    "gemini-2.5-pro": (1.25, 10.00),       # prompts up to 200k tokens
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.40),
}

# English prose and JSON run close to four characters a token on Gemini's
# tokenizer; it is an estimate, said as one in every report.
CHARS_PER_TOKEN = 4.0


def tokens_for(chars: int) -> int:
    return int(math.ceil(max(0, chars) / CHARS_PER_TOKEN))


def parse_price_overrides(values: Iterable[str]) -> Dict[str, Tuple[float, float]]:
    """--price MODEL=INPUT/OUTPUT (USD per 1M tokens), any number of times."""
    out: Dict[str, Tuple[float, float]] = {}
    for value in values or []:
        model, _, pair = str(value).partition("=")
        given_in, _, given_out = pair.partition("/")
        if not model or not given_in or not given_out:
            raise ValueError(f"--price takes MODEL=INPUT/OUTPUT, not {value!r}")
        out[model.strip()] = (float(given_in), float(given_out))
    return out


def price_of(model: str, overrides: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[float, float]:
    """The model's (input, output) price. A model id with a version suffix
    (gemini-2.5-flash-001) is priced as its family."""
    table = {**PRICES, **(overrides or {})}
    if model in table:
        return table[model]
    for name in sorted(table, key=len, reverse=True):
        if model.startswith(name + "-"):
            return table[name]
    raise KeyError(f"No list price for {model!r}: pass --price {model}=INPUT/OUTPUT (USD per 1M tokens).")


class Step:
    """One kind of model call: how many times, which model, the input
    tokens per call, and the output per call (low, cap)."""

    def __init__(self, name: str, model: str, calls: float, input_tokens: int, output_low: int, output_cap: int, note: str = "") -> None:
        self.name = name
        self.model = model
        self.calls = calls
        self.input_tokens = input_tokens
        self.output_low = output_low
        self.output_cap = output_cap
        self.note = note

    def cost(self, overrides: Optional[Dict[str, Tuple[float, float]]] = None) -> Tuple[float, float]:
        price_in, price_out = price_of(self.model, overrides)
        base = self.calls * self.input_tokens * price_in / 1_000_000
        return (
            base + self.calls * self.output_low * price_out / 1_000_000,
            base + self.calls * self.output_cap * price_out / 1_000_000,
        )

    def arithmetic(self, overrides: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
        price_in, price_out = price_of(self.model, overrides)
        low, high = self.cost(overrides)
        calls = f"{self.calls:g}"
        return (
            f"{self.name} [{self.model}]: {calls} calls x ({self.input_tokens:,} in x ${price_in:.2f}/M"
            f" + {self.output_low:,}..{self.output_cap:,} out x ${price_out:.2f}/M)"
            f" = ${low:.4f}..${high:.4f}{f'  ({self.note})' if self.note else ''}"
        )


def report(title: str, steps: List[Step], overrides: Optional[Dict[str, Tuple[float, float]]] = None) -> str:
    lines = [title]
    total_low = total_high = 0.0
    for step in steps:
        if step.calls <= 0:
            continue
        low, high = step.cost(overrides)
        total_low += low
        total_high += high
        lines.append("  " + step.arithmetic(overrides))
    lines.append(f"  Total: ${total_low:.2f} to ${total_high:.2f}")
    lines.append(
        f"  Prices USD per 1M tokens, list, checked {PRICES_CHECKED_ON} ({PRICES_SOURCE}); "
        f"input tokens = prompt characters / {CHARS_PER_TOKEN:g}; output low = the expected visible answer, "
        "high = the call's output cap (thinking is billed as output)."
    )
    return "\n".join(lines)
