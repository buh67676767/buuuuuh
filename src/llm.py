"""Turns plain-English strategy descriptions (or requests for new ideas) into
StrategySpec objects via the Claude API.

The model never writes code -- it fills in the typed StrategySpec schema
(see spec_strategy.py), so a generated strategy is only ever a composition of
the whitelisted indicators/operators the compiler already knows how to run.
"""

from __future__ import annotations

import os

import anthropic

from .spec_strategy import StrategyIdeas, StrategySpec

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """\
You are a quantitative trading strategy designer. You translate ideas into a
StrategySpec: a small typed structure of indicators and comparison rules --
never code.

Available indicator types (declare each with a unique short "id"):
- sma(period): simple moving average of close
- ema(period): exponential moving average of close
- rsi(period): Relative Strength Index, 0-100
- macd_line / macd_signal / macd_hist (fast, slow, signal): MACD family
- bb_upper / bb_mid / bb_lower (period, num_std): Bollinger Bands
- atr(period): Average True Range (volatility)
- roc(period): rate of change of close, in percent
- stoch_k(period): stochastic %K, 0-100

You may also reference the raw price fields directly without declaring an
indicator: open, high, low, close, volume.

Rules (entry_long / exit_long / entry_short / exit_short) are a list of up to
3 conditions, ALL of which must be true (AND) for the rule to fire. Each
condition compares `left` (an indicator id or price field) against either
`right_indicator` (another indicator id or price field) or `right_value` (a
constant number) using one of: gt, lt, gte, lte, cross_above, cross_below.
cross_above/cross_below fire only on the bar the relationship flips, so use
them for moving-average crossovers; use gt/lt for threshold conditions like
RSI oversold.

Only set allow_short=true if you also provide entry_short and exit_short.
Keep strategies simple, robust, and expressible purely with these primitives
-- do not invent indicator types or fields that aren't listed above.
"""


def _client(api_key: str | None = None) -> anthropic.Anthropic:
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "No Anthropic API key found. Paste one in the sidebar, or set "
            "ANTHROPIC_API_KEY in your .env file."
        )
    return anthropic.Anthropic(api_key=key)


def generate_strategy_from_description(description: str, api_key: str | None = None) -> StrategySpec:
    client = _client(api_key)
    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Translate this trading idea into a StrategySpec:\n\n{description}",
            }
        ],
        output_format=StrategySpec,
    )
    return response.parsed_output


def suggest_strategy_ideas(n: int = 3, context: str = "", api_key: str | None = None) -> list[StrategySpec]:
    client = _client(api_key)
    prompt = (
        f"Suggest {n} diverse, reasonable trading strategy ideas using the available "
        "primitives. Favor ideas that are meaningfully different from each other "
        "(e.g. trend-following vs. mean-reversion vs. volatility breakout)."
    )
    if context:
        prompt += f"\n\nContext from the user: {context}"

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_format=StrategyIdeas,
    )
    return response.parsed_output.ideas
