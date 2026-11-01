"""Trading Lab -- interactive strategy backtesting.

Run with: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from src import llm, strategy_store
from src.data import load_ohlcv
from src.engine import run_backtest
from src.metrics import metrics_table, rolling_window_table
from src.spec_strategy import SpecStrategy, StrategySpec
from src.strategies import STRATEGY_REGISTRY

load_dotenv()

st.set_page_config(page_title="Trading Lab", layout="wide")

# Fixed categorical order -- see dataviz skill: never cycle hues, assign by entity.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

TICKER_PRESETS = {
    "Stocks / ETFs": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "SPY", "QQQ", "VTI"],
    "Crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "DOGE-USD"],
    "Commodities": ["GC=F (Gold)", "SI=F (Silver)", "CL=F (Crude Oil)", "NG=F (Natural Gas)", "HG=F (Copper)"],
}

BUILTIN_DESCRIPTIONS = {
    "buy_and_hold": (
        "Buys at the start and holds for the whole period. The baseline every other "
        "strategy should beat -- if a strategy can't outperform this, its added "
        "complexity and trading costs aren't paying for themselves."
    ),
    "sma_cross": (
        "Trend-following. Goes long when the fast moving average rises above the slow "
        "one (a sign an uptrend is taking hold) and exits when it crosses back below. "
        "Aims to ride sustained trends and sit out choppy, directionless periods."
    ),
    "rsi_mean_reversion": (
        "Mean-reversion. Buys when RSI shows the price has dropped sharply (oversold), "
        "betting on a bounce back toward the average, and exits once momentum has "
        "recovered."
    ),
    "bollinger_breakout": (
        "Volatility breakout. Buys when price pushes above its upper Bollinger Band -- "
        "a statistically unusual move suggesting strong momentum -- and exits once "
        "price falls back toward the average."
    ),
}

BUILTIN_PARAMS = {
    "buy_and_hold": {},
    "sma_cross": {
        "fast": {"min": 5, "max": 100, "default": 20},
        "slow": {"min": 10, "max": 300, "default": 50},
    },
    "rsi_mean_reversion": {
        "period": {"min": 5, "max": 50, "default": 14},
        "oversold": {"min": 10, "max": 45, "default": 30},
        "exit_level": {"min": 40, "max": 70, "default": 50},
    },
    "bollinger_breakout": {
        "period": {"min": 5, "max": 100, "default": 20},
        "num_std": {"min": 0.5, "max": 4.0, "default": 2.0, "step": 0.1},
    },
}

RISK_WINDOWS = {
    "~1 month (21d)": 21,
    "~3 months (63d)": 63,
    "~6 months (126d)": 126,
    "~1 year (252d)": 252,
}


def init_state() -> None:
    defaults = {
        "api_key": "",
        "generated_spec": None,
        "suggested_ideas": [],
        "results": None,
        "results_meta": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def render_spec_card(spec: StrategySpec) -> None:
    st.markdown(f"**{spec.name}**")
    st.write(spec.description)
    if spec.indicators:
        st.caption("Indicators: " + ", ".join(f"`{i.id}` ({i.type})" for i in spec.indicators))

    def fmt_rule(rule_set):
        if rule_set is None:
            return "—"
        parts = []
        for c in rule_set.conditions:
            rhs = c.right_indicator if c.right_indicator is not None else c.right_value
            parts.append(f"{c.left} {c.operator} {rhs}")
        return " AND ".join(parts)

    st.write(f"Entry long: `{fmt_rule(spec.entry_long)}`")
    st.write(f"Exit long: `{fmt_rule(spec.exit_long)}`")
    if spec.allow_short:
        st.write(f"Entry short: `{fmt_rule(spec.entry_short)}`")
        st.write(f"Exit short: `{fmt_rule(spec.exit_short)}`")
    with st.expander("Raw spec (JSON)"):
        st.json(spec.model_dump())


init_state()

st.title("Trading Lab")
st.caption(
    "Describe strategies in plain English, or ask for ideas. Test them against real "
    "historical data with a no-lookahead backtest engine, and compare profit/loss and risk."
)

with st.sidebar:
    st.header("Market")
    asset_class = st.selectbox("Asset class", list(TICKER_PRESETS.keys()))
    preset = st.selectbox("Ticker", TICKER_PRESETS[asset_class] + ["Custom..."])
    if preset == "Custom...":
        ticker = st.text_input("Custom ticker (Yahoo Finance symbol)", value="AAPL").strip().upper()
    else:
        ticker = preset.split(" ")[0]

    st.header("Timeframe")
    c1, c2 = st.columns(2)
    start_date = c1.date_input("Start", value=(pd.Timestamp.today() - pd.DateOffset(years=5)).date())
    end_date = c2.date_input("End", value=pd.Timestamp.today().date())
    interval = st.selectbox("Interval", ["1d", "1wk"], index=0)

    st.header("Backtest settings")
    initial_cash = st.number_input("Initial cash ($)", min_value=100.0, value=10_000.0, step=1_000.0)
    commission_pct = st.number_input("Commission (%)", min_value=0.0, value=0.1, step=0.05) / 100
    slippage_pct = st.number_input("Slippage (%)", min_value=0.0, value=0.05, step=0.05) / 100

    st.header("Claude API key")
    st.session_state.api_key = st.text_input(
        "Anthropic API key",
        value=st.session_state.api_key,
        type="password",
        help=(
            "Used only in this session to call Claude for 'Describe a Strategy' and "
            "'Suggest Ideas'. Not written to disk. Alternatively, set ANTHROPIC_API_KEY "
            "in this project's .env file."
        ),
    )

tab_library, tab_describe, tab_suggest, tab_run = st.tabs(
    ["Strategy Library", "Describe a Strategy", "Suggest Ideas", "Run & Compare"]
)

with tab_library:
    st.subheader("Built-in strategies")
    for name in STRATEGY_REGISTRY:
        with st.expander(name, expanded=(name == "buy_and_hold")):
            st.caption(BUILTIN_DESCRIPTIONS.get(name, ""))
            st.checkbox("Include in next run", value=(name == "buy_and_hold"), key=f"chk_{name}")
            for param, cfg in BUILTIN_PARAMS.get(name, {}).items():
                if isinstance(cfg["default"], float):
                    st.number_input(
                        param,
                        min_value=float(cfg["min"]),
                        max_value=float(cfg["max"]),
                        value=float(cfg["default"]),
                        step=float(cfg.get("step", 0.1)),
                        key=f"param_{name}_{param}",
                    )
                else:
                    st.number_input(
                        param,
                        min_value=int(cfg["min"]),
                        max_value=int(cfg["max"]),
                        value=int(cfg["default"]),
                        step=1,
                        key=f"param_{name}_{param}",
                    )

    st.subheader("Your saved strategies")
    saved = strategy_store.load_specs()
    if not saved:
        st.caption("None yet -- describe one or ask for ideas in the other tabs.")
    for slug, spec in saved:
        with st.expander(spec.name):
            st.checkbox("Include in next run", value=True, key=f"chk_custom_{slug}")
            render_spec_card(spec)
            if st.button("Delete", key=f"delete_{slug}"):
                strategy_store.delete_spec(slug)
                st.rerun()

with tab_describe:
    st.subheader("Describe a strategy in plain English")
    description = st.text_area(
        "Strategy idea",
        height=120,
        placeholder=(
            "e.g. Buy when the 50-day moving average crosses above the 200-day moving "
            "average, and sell when RSI goes above 75."
        ),
    )
    if st.button("Generate strategy", type="primary"):
        if not description.strip():
            st.warning("Describe a strategy idea first.")
        else:
            with st.spinner("Asking Claude to design the strategy..."):
                try:
                    st.session_state.generated_spec = llm.generate_strategy_from_description(
                        description, api_key=st.session_state.api_key or None
                    )
                except Exception as e:
                    st.error(f"Couldn't generate a strategy: {e}")

    if st.session_state.generated_spec:
        spec = st.session_state.generated_spec
        render_spec_card(spec)
        save_name = st.text_input("Save as", value=spec.name, key="save_name_describe")
        if st.button("Save to library", key="save_describe"):
            strategy_store.save_spec(spec.model_copy(update={"name": save_name}))
            st.success(f"Saved '{save_name}' to the library.")
            st.session_state.generated_spec = None
            st.rerun()

with tab_suggest:
    st.subheader("Get new strategy ideas from Claude")
    n_ideas = st.slider("How many ideas?", 1, 5, 3)
    context = st.text_input("Optional context (e.g. 'low-volatility ETFs', 'crypto momentum')", "")
    if st.button("Suggest ideas", type="primary"):
        with st.spinner("Asking Claude for ideas..."):
            try:
                st.session_state.suggested_ideas = llm.suggest_strategy_ideas(
                    n_ideas, context, api_key=st.session_state.api_key or None
                )
            except Exception as e:
                st.error(f"Couldn't get ideas: {e}")

    for idx, spec in enumerate(st.session_state.suggested_ideas):
        with st.container(border=True):
            render_spec_card(spec)
            if st.button("Save to library", key=f"save_idea_{idx}"):
                strategy_store.save_spec(spec)
                st.success(f"Saved '{spec.name}' to the library.")

with tab_run:
    st.subheader("Run backtest")
    window_label = st.selectbox("Risk window (for the variance table)", list(RISK_WINDOWS.keys()), index=2)
    window_days = RISK_WINDOWS[window_label]

    if st.button("Run backtest", type="primary"):
        strategies = []
        for name, factory in STRATEGY_REGISTRY.items():
            if st.session_state.get(f"chk_{name}"):
                kwargs = {
                    param: st.session_state.get(f"param_{name}_{param}")
                    for param in BUILTIN_PARAMS.get(name, {})
                }
                strategies.append(factory(**kwargs))

        for slug, spec in strategy_store.load_specs():
            if st.session_state.get(f"chk_custom_{slug}"):
                try:
                    strategies.append(SpecStrategy(spec))
                except Exception as e:
                    st.error(f"Skipping '{spec.name}': {e}")

        if not strategies:
            st.warning("Select at least one strategy in the Strategy Library tab.")
        elif not ticker:
            st.warning("Enter a ticker in the sidebar.")
        else:
            df = None
            try:
                with st.spinner(f"Loading {ticker} data..."):
                    df = load_ohlcv(ticker, str(start_date), str(end_date), interval)
            except Exception as e:
                st.error(f"Couldn't load data for {ticker}: {e}")

            if df is not None:
                results = [
                    run_backtest(
                        df,
                        strat,
                        initial_cash=initial_cash,
                        commission_pct=commission_pct,
                        slippage_pct=slippage_pct,
                    )
                    for strat in strategies
                ]
                st.session_state.results = results
                st.session_state.results_meta = {
                    "ticker": ticker,
                    "start": str(start_date),
                    "end": str(end_date),
                    "interval": interval,
                    "window_days": window_days,
                }

    results = st.session_state.results
    meta = st.session_state.results_meta
    if results and meta:
        st.caption(
            f"{meta['ticker']} — {len(results)} strategies, {meta['start']} to {meta['end']} "
            f"({meta['interval']})"
        )

        perf = metrics_table(results)

        st.subheader("Profit / loss")
        cols = st.columns(min(len(results), 4))
        for i, res in enumerate(results):
            row = perf.iloc[i]
            with cols[i % len(cols)]:
                st.metric(
                    res.strategy_name,
                    f"{row['total_return_pct']:.1f}%",
                    f"CAGR {row['cagr_pct']:.1f}%/yr",
                )

        display_cols = {
            "total_return_pct": "Total return %",
            "cagr_pct": "CAGR %/yr",
            "annualized_vol_pct": "Volatility %/yr",
            "sharpe": "Sharpe",
            "max_drawdown_pct": "Max drawdown %",
            "win_rate_pct": "Win rate %",
            "profit_factor": "Profit factor",
            "num_round_trips": "Round trips",
        }
        st.dataframe(perf[list(display_cols)].rename(columns=display_cols), use_container_width=True)

        st.subheader("Risk — result variance across start dates")
        st.caption(
            f"Return over every rolling {meta['window_days']}-bar window. A wide spread "
            "between best/worst means the outcome depends heavily on when you started, "
            "not just on average performance."
        )
        risk = rolling_window_table(results, meta["window_days"])
        risk_cols = {
            "mean_pct": "Mean %",
            "std_pct": "Std dev %",
            "best_pct": "Best window %",
            "worst_pct": "Worst window %",
            "n_windows": "# windows",
        }
        st.dataframe(risk[list(risk_cols)].rename(columns=risk_cols), use_container_width=True)

        st.subheader("Equity curves")
        fig = go.Figure()
        for i, res in enumerate(results):
            fig.add_trace(
                go.Scatter(
                    x=res.equity_curve.index,
                    y=res.equity_curve.values,
                    mode="lines",
                    name=res.strategy_name,
                    line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                )
            )
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Equity ($)",
            legend_title="Strategy",
            height=450,
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Total return by strategy")
        bar_fig = go.Figure(
            go.Bar(
                x=[res.strategy_name for res in results],
                y=perf["total_return_pct"].tolist(),
                marker_color=[PALETTE[i % len(PALETTE)] for i in range(len(results))],
            )
        )
        bar_fig.update_layout(yaxis_title="Total return (%)", height=350)
        st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.info("Pick strategies in the Strategy Library tab, then run a backtest.")
