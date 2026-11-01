"""A JSON-describable strategy format, and a safe compiler from spec -> Strategy.

This is what makes LLM-generated strategies safe to run: the LLM never produces
Python code. It produces a `StrategySpec` -- a small typed tree of indicator
declarations and comparison rules, built entirely from whitelisted enums,
numbers, and ids. `SpecStrategy` interprets that tree with a fixed set of
functions (no `eval`/`exec` anywhere), so even an adversarial or hallucinated
spec can only ever compose the operations defined below.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from . import indicators as ind
from .strategies import Strategy

PRICE_FIELDS = {"open", "high", "low", "close", "volume"}

IndicatorType = Literal[
    "sma", "ema", "rsi",
    "macd_line", "macd_signal", "macd_hist",
    "bb_upper", "bb_mid", "bb_lower",
    "atr", "roc", "stoch_k",
]

Operator = Literal["gt", "lt", "gte", "lte", "cross_above", "cross_below"]


class IndicatorSpec(BaseModel):
    id: str = Field(..., min_length=1, max_length=40, description="Unique short id, e.g. 'sma_fast'")
    type: IndicatorType
    period: int = Field(14, ge=2, le=300)
    fast: int = Field(12, ge=2, le=200)
    slow: int = Field(26, ge=3, le=300)
    signal: int = Field(9, ge=2, le=100)
    num_std: float = Field(2.0, ge=0.5, le=5.0)


class Condition(BaseModel):
    left: str = Field(..., description="Indicator id or price field (open/high/low/close/volume)")
    operator: Operator
    right_indicator: Optional[str] = Field(
        None, description="Indicator id or price field to compare against"
    )
    right_value: Optional[float] = Field(
        None, description="Constant to compare against (used instead of right_indicator)"
    )


class RuleSet(BaseModel):
    conditions: List[Condition] = Field(
        ..., min_length=1, max_length=3, description="All conditions must hold (AND) for this rule to fire"
    )


class StrategySpec(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    description: str = Field(..., min_length=1, max_length=600)
    indicators: List[IndicatorSpec] = Field(default_factory=list, max_length=6)
    allow_short: bool = False
    entry_long: RuleSet
    exit_long: RuleSet
    entry_short: Optional[RuleSet] = None
    exit_short: Optional[RuleSet] = None


class StrategyIdeas(BaseModel):
    ideas: List[StrategySpec] = Field(..., min_length=1, max_length=5)


def _compute_indicator_series(df: pd.DataFrame, spec: IndicatorSpec) -> pd.Series:
    t = spec.type
    if t == "sma":
        return ind.sma(df["close"], spec.period)
    if t == "ema":
        return ind.ema(df["close"], spec.period)
    if t == "rsi":
        return ind.rsi(df["close"], spec.period)
    if t in ("macd_line", "macd_signal", "macd_hist"):
        line, sig, hist = ind.macd(df["close"], spec.fast, spec.slow, spec.signal)
        return {"macd_line": line, "macd_signal": sig, "macd_hist": hist}[t]
    if t in ("bb_upper", "bb_mid", "bb_lower"):
        upper, mid, lower = ind.bollinger(df["close"], spec.period, spec.num_std)
        return {"bb_upper": upper, "bb_mid": mid, "bb_lower": lower}[t]
    if t == "atr":
        return ind.atr(df, spec.period)
    if t == "roc":
        return ind.roc(df["close"], spec.period)
    if t == "stoch_k":
        return ind.stoch_k(df, spec.period)
    raise ValueError(f"Unknown indicator type: {t}")  # unreachable: Literal-guarded


def build_series_context(df: pd.DataFrame, indicator_specs: List[IndicatorSpec]) -> dict:
    ctx = {f: df[f] for f in PRICE_FIELDS}
    for spec in indicator_specs:
        ctx[spec.id] = _compute_indicator_series(df, spec)
    return ctx


def _resolve(name: str, ctx: dict) -> pd.Series:
    if name not in ctx:
        raise ValueError(f"Unknown reference '{name}' -- not a declared indicator id or price field")
    return ctx[name]


def _eval_condition(cond: Condition, ctx: dict) -> pd.Series:
    left = _resolve(cond.left, ctx)
    if cond.right_indicator is not None:
        right = _resolve(cond.right_indicator, ctx)
    elif cond.right_value is not None:
        right = cond.right_value
    else:
        raise ValueError("Condition must set right_indicator or right_value")

    op = cond.operator
    if op == "gt":
        return left > right
    if op == "lt":
        return left < right
    if op == "gte":
        return left >= right
    if op == "lte":
        return left <= right
    if op == "cross_above":
        prev_left = left.shift(1)
        prev_right = right.shift(1) if isinstance(right, pd.Series) else right
        return (left > right) & (prev_left <= prev_right)
    if op == "cross_below":
        prev_left = left.shift(1)
        prev_right = right.shift(1) if isinstance(right, pd.Series) else right
        return (left < right) & (prev_left >= prev_right)
    raise ValueError(f"Unknown operator '{op}'")  # unreachable: Literal-guarded


def _eval_ruleset(ruleset: RuleSet, ctx: dict) -> pd.Series:
    result = None
    for cond in ruleset.conditions:
        c = _eval_condition(cond, ctx).fillna(False)
        result = c if result is None else (result & c)
    return result


def validate_spec(spec: StrategySpec, max_indicators: int = 6, max_conditions: int = 3) -> None:
    if len(spec.indicators) > max_indicators:
        raise ValueError(f"Too many indicators (max {max_indicators})")

    known_ids = set(PRICE_FIELDS)
    for indicator in spec.indicators:
        if indicator.id in known_ids:
            raise ValueError(f"Indicator id '{indicator.id}' collides with a reserved name or is duplicated")
        known_ids.add(indicator.id)

    def check_ruleset(rule_set: Optional[RuleSet], label: str) -> None:
        if rule_set is None:
            return
        if len(rule_set.conditions) > max_conditions:
            raise ValueError(f"{label}: too many conditions (max {max_conditions})")
        for cond in rule_set.conditions:
            if cond.left not in known_ids:
                raise ValueError(f"{label}: unknown reference '{cond.left}'")
            if cond.right_indicator is not None and cond.right_indicator not in known_ids:
                raise ValueError(f"{label}: unknown reference '{cond.right_indicator}'")
            if cond.right_indicator is None and cond.right_value is None:
                raise ValueError(f"{label}: condition needs right_indicator or right_value")

    check_ruleset(spec.entry_long, "entry_long")
    check_ruleset(spec.exit_long, "exit_long")
    check_ruleset(spec.entry_short, "entry_short")
    check_ruleset(spec.exit_short, "exit_short")

    if spec.allow_short and (spec.entry_short is None or spec.exit_short is None):
        raise ValueError("allow_short is true but entry_short/exit_short rules are missing")


class SpecStrategy(Strategy):
    """Compiles a validated StrategySpec into a runnable Strategy."""

    def __init__(self, spec: StrategySpec):
        validate_spec(spec)
        self.spec = spec
        self.name = spec.name

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        spec = self.spec
        ctx = build_series_context(df, spec.indicators)

        entry_long = _eval_ruleset(spec.entry_long, ctx).to_numpy()
        exit_long = _eval_ruleset(spec.exit_long, ctx).to_numpy()
        entry_short = _eval_ruleset(spec.entry_short, ctx).to_numpy() if spec.entry_short else None
        exit_short = _eval_ruleset(spec.exit_short, ctx).to_numpy() if spec.exit_short else None

        position = 0
        out = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            if position == 0:
                if entry_long[i]:
                    position = 1
                elif spec.allow_short and entry_short is not None and entry_short[i]:
                    position = -1
            elif position == 1:
                if exit_long[i]:
                    position = 0
            elif position == -1:
                if exit_short is not None and exit_short[i]:
                    position = 0
            out[i] = position
        return pd.Series(out, index=df.index)
