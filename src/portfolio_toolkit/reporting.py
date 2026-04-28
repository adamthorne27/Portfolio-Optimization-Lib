from __future__ import annotations

import json
from html import escape
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import quantstats as qs

from .contracts import BacktestResult


def _annualize_total_return(total_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    if total_return <= -1.0:
        return -1.0
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _evaluation_start(result: BacktestResult) -> pd.Timestamp | None:
    if result.weights.empty:
        return None
    return pd.Timestamp(result.weights.index.min())


def _slice_from_start(series: pd.Series, start: pd.Timestamp | None) -> pd.Series:
    series = series.sort_index()
    if start is None:
        return series
    return series.loc[series.index >= start]


def _evaluation_years(index: pd.Index) -> float:
    if len(index) < 2:
        return 0.0
    start = pd.Timestamp(index.min())
    end = pd.Timestamp(index.max())
    elapsed_days = max((end - start).days, 0)
    return float(elapsed_days / 365.25)


def _preferred_benchmark(result: BacktestResult) -> tuple[str | None, pd.Series | None]:
    if result.benchmark_returns.empty:
        return None, None
    if "SPY" in result.benchmark_returns.columns:
        benchmark_name = "SPY"
    else:
        benchmark_name = str(result.benchmark_returns.columns[0])
    return benchmark_name, result.benchmark_returns[benchmark_name].astype(float)


def _add_quantstats_alpha_note(report_path: Path, benchmark_name: str | None) -> None:
    if benchmark_name is None:
        return

    safe_benchmark = escape(benchmark_name)
    note = f"""
    <div class="container" style="border:1px solid #b7c9d9;background:#f6fbff;padding:12px 14px;margin:0 auto 22px;">
        <strong>Portfolio Toolkit note:</strong>
        QuantStats' <code>Alpha</code> row is annualized regression alpha vs <code>{safe_benchmark}</code>
        using realized daily strategy returns. It is not the model's <code>expected_alpha</code> prediction,
        not a <code>forward_alpha_*</code> training target, and not the simple total-return difference.
        For simple total-return difference, use <code>excess_return_vs_benchmark</code> in <code>metrics.json</code>.
    </div>
"""
    html = report_path.read_text(encoding="utf-8")
    html = html.replace("<tr><td>Alpha</td>", f"<tr><td>QuantStats Alpha vs {safe_benchmark}</td>", 1)
    if "<body onload=\"save()\">" in html:
        html = html.replace("<body onload=\"save()\">", f"<body onload=\"save()\">\n{note}", 1)
    elif "<body>" in html:
        html = html.replace("<body>", f"<body>\n{note}", 1)
    else:
        html = f"{note}\n{html}"
    report_path.write_text(html, encoding="utf-8")


def build_metrics(result: BacktestResult) -> dict[str, float]:
    evaluation_start = _evaluation_start(result)
    nav = _slice_from_start(result.nav.astype(float), evaluation_start)
    returns = _slice_from_start(result.returns.astype(float), evaluation_start)
    turnover = _slice_from_start(result.turnover.astype(float), evaluation_start)
    if nav.empty:
        nav = result.nav.astype(float)
    if returns.empty:
        returns = result.returns.astype(float)

    years = _evaluation_years(nav.index)
    periods_per_year = float(len(returns) / years) if years > 0 and len(returns) else 0.0
    total_return = float(nav.iloc[-1] / nav.iloc[0] - 1.0) if len(nav) > 1 else 0.0
    annual_return = _annualize_total_return(total_return, years)
    annual_volatility = float(returns.std(ddof=0) * sqrt(periods_per_year)) if periods_per_year > 0 else 0.0
    downside = returns[returns < 0.0]
    downside_vol = float(downside.std(ddof=0) * sqrt(periods_per_year)) if periods_per_year > 0 and len(downside) else 0.0
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0
    metrics = {
        "evaluation_years": years,
        "evaluation_trading_days": float(len(returns)),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_volatility,
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar": float(calmar),
        "average_turnover": float(turnover.mean()) if len(turnover) else 0.0,
    }
    benchmark_total_return = 0.0
    benchmark_annual_return = 0.0
    benchmark_annual_volatility = 0.0
    benchmark_sharpe = 0.0
    benchmark_max_drawdown = 0.0
    excess_return = total_return
    _, benchmark_returns = _preferred_benchmark(result)
    if benchmark_returns is not None:
        benchmark_returns = _slice_from_start(benchmark_returns, evaluation_start)
        benchmark_nav = (1.0 + benchmark_returns).cumprod()
        benchmark_total_return = float(benchmark_nav.iloc[-1] - 1.0) if len(benchmark_nav) else 0.0
        benchmark_annual_return = _annualize_total_return(benchmark_total_return, years)
        benchmark_annual_volatility = (
            float(benchmark_returns.std(ddof=0) * sqrt(periods_per_year)) if periods_per_year > 0 else 0.0
        )
        benchmark_sharpe = (
            benchmark_annual_return / benchmark_annual_volatility if benchmark_annual_volatility > 0 else 0.0
        )
        benchmark_drawdown = benchmark_nav / benchmark_nav.cummax() - 1.0
        benchmark_max_drawdown = float(benchmark_drawdown.min()) if len(benchmark_drawdown) else 0.0
        excess_return = total_return - benchmark_total_return
    metrics["benchmark_total_return"] = benchmark_total_return
    metrics["benchmark_annual_return"] = benchmark_annual_return
    metrics["benchmark_annual_volatility"] = benchmark_annual_volatility
    metrics["benchmark_sharpe"] = benchmark_sharpe
    metrics["benchmark_max_drawdown"] = benchmark_max_drawdown
    metrics["excess_return_vs_benchmark"] = excess_return
    metrics["annual_excess_return_vs_benchmark"] = annual_return - benchmark_annual_return
    metrics["sharpe_vs_benchmark"] = sharpe - benchmark_sharpe
    return metrics


def write_quantstats_report(result: BacktestResult, output_dir: str | Path) -> Path:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "quantstats.html"
    benchmark_name, benchmark = _preferred_benchmark(result)
    evaluation_start = _evaluation_start(result)
    returns = _slice_from_start(result.returns.astype(float), evaluation_start)
    if benchmark is not None:
        benchmark = _slice_from_start(benchmark, evaluation_start)
    years = _evaluation_years(returns.index)
    periods_per_year = float(len(returns) / years) if years > 0 and len(returns) else 252.0
    qs.reports.html(
        returns,
        benchmark=benchmark,
        output=str(report_path),
        title=f"{result.strategy_name} tear sheet",
        periods_per_year=periods_per_year,
    )
    _add_quantstats_alpha_note(report_path, benchmark_name)
    return report_path


def write_backtest_artifacts(result: BacktestResult, output_dir: str | Path) -> dict[str, str]:
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    weights_path = output_path / "weights.parquet"
    nav_path = output_path / "nav.parquet"
    returns_path = output_path / "returns.parquet"
    turnover_path = output_path / "turnover.parquet"
    benchmarks_path = output_path / "benchmarks.parquet"
    metrics_path = output_path / "metrics.json"
    metrics_table_path = output_path / "metrics_table.parquet"
    result.weights.to_parquet(weights_path)
    result.nav.to_frame(name="nav").to_parquet(nav_path)
    result.returns.to_frame(name="returns").to_parquet(returns_path)
    result.turnover.to_frame(name="turnover").to_parquet(turnover_path)
    result.benchmark_returns.to_parquet(benchmarks_path)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(result.metrics, handle, indent=2, sort_keys=True)
    pd.DataFrame([{"metric": key, "value": value} for key, value in sorted(result.metrics.items())]).to_parquet(metrics_table_path, index=False)
    report_path = write_quantstats_report(result, output_path)
    artifact_paths = {
        "weights": str(weights_path),
        "nav": str(nav_path),
        "returns": str(returns_path),
        "turnover": str(turnover_path),
        "benchmarks": str(benchmarks_path),
        "metrics": str(metrics_path),
        "metrics_table": str(metrics_table_path),
        "quantstats_report": str(report_path),
    }
    result.artifact_paths.update(artifact_paths)
    return artifact_paths
