from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static" / "eamon_dojo"
DEFAULT_PORT = 8787


LESSONS: list[dict[str, Any]] = [
    {
        "id": "returns-panel",
        "title": "1. Price Panel Basics",
        "stage": "Prerequisite",
        "source": "Barebones setup before Eamon's Week 1 XGBoost notebook",
        "prerequisites": [
            "A long-form price frame has one row per date/ticker.",
            "Use `pivot(index='date', columns='ticker', values='adj_close')` to make a wide price matrix.",
            "Use `pct_change()` to turn prices into daily returns.",
            "Never compute returns by mixing tickers in long-form order.",
        ],
        "problem": "Implement `make_return_matrix(prices)` so it returns a wide daily return matrix indexed by date and columned by ticker. Sort dates, use adjusted close, and fill the first return row with `0.0`.",
        "starter_code": """\
import pandas as pd

def make_return_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    # TODO: pivot adjusted close prices into a wide matrix.
    # TODO: compute percent changes by ticker.
    # TODO: fill the first row with 0.0 and return the frame.
    pass
""",
    },
    {
        "id": "features-target",
        "title": "2. Features And Forward Target",
        "stage": "Week 1 XGBoost",
        "source": "MODELS/Eamon/xgboost_model_wk1.ipynb",
        "prerequisites": [
            "XGBoost needs a numeric feature matrix `X` and a target vector `y`.",
            "A forward-return target uses future prices: `price.shift(-horizon) / price - 1`.",
            "Features must only use information available at or before that row's date.",
            "Momentum is usually a past percent change. Volatility is a rolling standard deviation of returns.",
        ],
        "problem": "Implement `build_basic_feature_target(prices, horizon)` with columns `date`, `ticker`, `momentum_5d`, `vol_5d`, and `forward_return`. Drop rows where any of those three numeric columns is missing.",
        "starter_code": """\
import pandas as pd

def build_basic_feature_target(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    # TODO: create a sorted wide adjusted-close matrix.
    # TODO: compute daily returns.
    # TODO: create momentum_5d, vol_5d, and forward_return.
    # TODO: return a clean long-form frame.
    pass
""",
    },
    {
        "id": "xgb-classifier",
        "title": "3. First XGBoost Classifier",
        "stage": "Week 1 XGBoost",
        "source": "MODELS/Eamon/xgboost_model_wk1.ipynb",
        "prerequisites": [
            "XGBoost is gradient boosted decision trees: many small trees are trained sequentially, and each new tree tries to fix the previous trees' mistakes.",
            "`XGBClassifier` predicts classes; `predict_proba` returns class probabilities.",
            "For a ranking portfolio, a simple target can be whether forward return is above the date median.",
            "Use a tiny number of estimators while learning so runs stay fast.",
            "The model score used by the portfolio can be the probability of class `1`.",
        ],
        "problem": "Implement `fit_tiny_xgb(panel)` using `momentum_5d` and `vol_5d`. Create `target` as `forward_return > date median`, train an `XGBClassifier`, and return `(model, predictions)` where predictions has `date`, `ticker`, and `expected_return`.",
        "starter_code": """\
import pandas as pd
from xgboost import XGBClassifier

FEATURES = ["momentum_5d", "vol_5d"]

def fit_tiny_xgb(panel: pd.DataFrame):
    # TODO: copy panel and create a binary target by date.
    # TODO: fit a small XGBClassifier with deterministic settings.
    # TODO: return the model and a prediction frame.
    pass
""",
    },
    {
        "id": "prediction-contract",
        "title": "4. Toolkit Prediction Contract",
        "stage": "Week 1 XGBoost",
        "source": "MODELS/Eamon/xgboost_model_wk1.ipynb",
        "prerequisites": [
            "The toolkit expects one prediction row per date/ticker/horizon.",
            "Required columns are `date`, `ticker`, `horizon`, and `expected_return`.",
            "`date` should be datetime-like and `ticker` should be uppercase.",
            "This table is the bridge between model inference and portfolio construction.",
        ],
        "problem": "Implement `make_prediction_contract(raw_predictions, horizon)` so it returns exactly `date`, `ticker`, `horizon`, `expected_return` in that order. Uppercase tickers and preserve the model score as `expected_return`.",
        "starter_code": """\
import pandas as pd

def make_prediction_contract(raw_predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # TODO: normalize date and ticker.
    # TODO: add horizon.
    # TODO: rename score to expected_return if needed.
    # TODO: return the required columns in order.
    pass
""",
    },
    {
        "id": "rank-weights",
        "title": "5. Rank Long-Only Weights",
        "stage": "Week 1 Portfolio Builder",
        "source": "MODELS/Eamon/xgboost_model_wk1.ipynb",
        "prerequisites": [
            "A long-only portfolio has non-negative weights that sum to `1.0` each rebalance date.",
            "Rank weighting gives the best score the largest weight, second best the next largest, and so on.",
            "Group by date so each rebalance is independent.",
            "This is the simple version before Eamon's min-variance backbone.",
        ],
        "problem": "Implement `rank_long_only_weights(predictions)`. For each date, sort descending by `expected_return`, assign raw scores `n, n-1, ..., 1`, normalize them to sum to 1, and return a wide weight frame.",
        "starter_code": """\
import numpy as np
import pandas as pd

def rank_long_only_weights(predictions: pd.DataFrame) -> pd.DataFrame:
    # TODO: build date-indexed, ticker-columned weights.
    pass
""",
    },
    {
        "id": "minvar-backbone",
        "title": "6. Minimum-Variance Backbone",
        "stage": "Min-Var + XGB Tilt",
        "source": "MODELS/Eamon/minvar_wk3.ipynb and eamon_xgboost_final.ipynb",
        "prerequisites": [
            "Minimum variance starts from a covariance matrix, not model predictions.",
            "The unconstrained long-only approximation can use inverse covariance times a vector of ones.",
            "Weights must be clipped to non-negative values and renormalized.",
            "Eamon's final notebook replaces this simple covariance with an EW FF5 factor covariance model.",
        ],
        "problem": "Implement `simple_minvar_weights(cov, tickers)`. Use the pseudo-inverse of the covariance matrix, multiply by a vector of ones, clip negative weights to zero, and normalize to sum to 1.",
        "starter_code": """\
import numpy as np
import pandas as pd

def simple_minvar_weights(cov: np.ndarray, tickers: list[str]) -> pd.Series:
    # TODO: compute inverse-covariance weights.
    # TODO: clip to long-only.
    # TODO: normalize and return a Series indexed by tickers.
    pass
""",
    },
    {
        "id": "xgb-tilt",
        "title": "7. Centered XGBoost Tilt",
        "stage": "Min-Var + XGB Tilt",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "Eamon's final model uses min-var as the backbone.",
            "XGBoost only gets a small tilt budget controlled by `TILT_ALPHA`.",
            "A centered signal subtracts the cross-sectional mean so the tilt is relative.",
            "After tilting, project back to long-only, capped, fully invested weights.",
        ],
        "problem": "Implement `apply_centered_tilt(minvar_weights, probabilities, tilt_alpha, max_weight)`. Center probabilities, scale their absolute exposure to `tilt_alpha`, add to min-var weights, clip to `[0, max_weight]`, and normalize.",
        "starter_code": """\
import pandas as pd

def apply_centered_tilt(
    minvar_weights: pd.Series,
    probabilities: pd.Series,
    tilt_alpha: float = 0.25,
    max_weight: float = 0.35,
) -> pd.Series:
    # TODO: align probabilities to minvar_weights.
    # TODO: center and scale the signal.
    # TODO: add the tilt, clip, normalize, and return.
    pass
""",
    },
    {
        "id": "no-leakage-rebalance",
        "title": "11. Rolling Rebalance Without Leakage",
        "stage": "Final Rolling Loop",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "A rebalance on date `t` may use features strictly before `t`.",
            "Use `searchsorted(..., side='left') - 1` to find the latest prior feature snapshot.",
            "Do not use same-day or future target data for model scoring.",
            "This rule is what makes the rolling backtest auditable.",
        ],
        "problem": "Implement `latest_feature_date_before(rebalance_date, feature_dates)`. Return the latest feature date strictly before the rebalance date. Raise `ValueError` if none exists.",
        "starter_code": """\
import pandas as pd

def latest_feature_date_before(rebalance_date, feature_dates) -> pd.Timestamp:
    # TODO: sort feature_dates as a DatetimeIndex.
    # TODO: find the latest date strictly before rebalance_date.
    # TODO: raise ValueError if no prior feature date exists.
    pass
""",
    },
    {
        "id": "vol-feature-pack",
        "title": "8. Volatility Feature Pack",
        "stage": "Final Feature Engineering",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "Eamon's final feature set adds volatility families beyond simple returns.",
            "Volatility of volatility measures how unstable recent volatility itself has been.",
            "Current implied volatility was used as a static ticker-level feature in the notebook, with a missing flag.",
            "Feature rows should stay long-form so they can merge into the toolkit feature panel.",
        ],
        "problem": "Implement `add_vol_feature_pack(feature_frame, prices_frame, iv_snapshot, window)`. Add `vol_of_vol`, `current_iv`, and `current_iv_missing` to the feature frame. `iv_snapshot` has `ticker` and `current_iv`; missing IV should become the cross-sectional median and set the missing flag to `1.0`.",
        "starter_code": """\
import pandas as pd

def add_vol_feature_pack(
    feature_frame: pd.DataFrame,
    prices_frame: pd.DataFrame,
    iv_snapshot: pd.DataFrame,
    window: int = 5,
) -> pd.DataFrame:
    # TODO: compute return volatility, then volatility-of-volatility.
    # TODO: merge vol_of_vol into feature_frame.
    # TODO: merge current_iv and create current_iv_missing.
    # TODO: fill missing current_iv with median current_iv.
    pass
""",
    },
    {
        "id": "ew-factor-weights",
        "title": "9. Exponentially Weighted Factor Window",
        "stage": "Final Risk Model",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "Eamon's final risk model gives recent observations more influence.",
            "A half-life controls how quickly older rows decay.",
            "The newest observation should receive the largest sample weight.",
            "Normalizing weights to mean `1.0` keeps regression scale stable.",
        ],
        "problem": "Implement `exponential_sample_weights(n, halflife)`. Return a NumPy array of length `n`, newest row last, with mean weight exactly `1.0` and positive exponentially decaying weights.",
        "starter_code": """\
import numpy as np

def exponential_sample_weights(n: int, halflife: float) -> np.ndarray:
    # TODO: validate n and halflife.
    # TODO: make oldest rows smaller and newest rows larger.
    # TODO: normalize to mean 1.0.
    pass
""",
    },
    {
        "id": "factor-covariance",
        "title": "10. FF5 Factor Covariance",
        "stage": "Final Risk Model",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "A factor risk model combines shared factor risk and ticker-specific residual risk.",
            "If `B` is ticker-by-factor beta and `F` is factor covariance, shared covariance is `B @ F @ B.T`.",
            "Residual variance is added only to the diagonal.",
            "The output must be ticker-indexed and ticker-columned for a min-variance optimizer.",
        ],
        "problem": "Implement `build_factor_covariance(beta, factor_cov, residual_vars)`. Return a DataFrame equal to `B @ F @ B.T + diag(residual_vars)`, preserving ticker order from `beta.index`.",
        "starter_code": """\
import numpy as np
import pandas as pd

def build_factor_covariance(
    beta: pd.DataFrame,
    factor_cov: pd.DataFrame,
    residual_vars: pd.Series,
) -> pd.DataFrame:
    # TODO: align factor_cov to beta columns.
    # TODO: compute shared factor covariance.
    # TODO: add residual variances to the diagonal.
    # TODO: return a ticker-indexed covariance DataFrame.
    pass
""",
    },
    {
        "id": "submission-manifest",
        "title": "12. Submission Manifest",
        "stage": "Packaging And MLflow",
        "source": "MODELS/Eamon/eamon_xgboost_final.ipynb",
        "prerequisites": [
            "The final notebook logs the XGBoost model artifact and metadata needed to reproduce it.",
            "A manifest should tell the evaluator feature order, horizon, rebalance frequency, and required functions.",
            "`model_config` describes how predictions become portfolio weights.",
            "This is the bridge between a working notebook and a fair backtest submission.",
        ],
        "problem": "Implement `build_submission_manifest(model_name, feature_names, horizon, rebalance_frequency)`. Return a dict with `model_name`, `model_family`, `feature_names`, `horizon`, `rebalance_frequency`, and `model_config` containing `estimator`, `portfolio_builder`, and `required_functions`.",
        "starter_code": """\
def build_submission_manifest(
    model_name: str,
    feature_names: list[str],
    horizon: int,
    rebalance_frequency: str,
) -> dict:
    # TODO: build the required manifest dictionary.
    pass
""",
    },
]


TESTS: dict[str, str] = {
    "returns-panel": """\
prices = pd.DataFrame({
    "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"]),
    "ticker": ["AAA", "AAA", "BBB", "BBB"],
    "adj_close": [100.0, 110.0, 50.0, 45.0],
})
out = make_return_matrix(prices)
assert list(out.columns) == ["AAA", "BBB"], out.columns
assert float(out.loc[pd.Timestamp("2024-01-01"), "AAA"]) == 0.0
assert round(float(out.loc[pd.Timestamp("2024-01-02"), "AAA"]), 6) == 0.1
assert round(float(out.loc[pd.Timestamp("2024-01-02"), "BBB"]), 6) == -0.1
print("passed: return matrix is wide, sorted, and ticker-safe")
""",
    "features-target": """\
dates = pd.date_range("2024-01-01", periods=9, freq="D")
prices = pd.DataFrame({
    "date": list(dates) * 2,
    "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
    "adj_close": [10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 19, 18, 17, 18, 19, 20, 21, 22],
})
out = build_basic_feature_target(prices, horizon=2)
assert {"date", "ticker", "momentum_5d", "vol_5d", "forward_return"}.issubset(out.columns)
assert out[["momentum_5d", "vol_5d", "forward_return"]].isna().sum().sum() == 0
aaa = out.loc[out["ticker"] == "AAA"].sort_values("date").iloc[0]
assert aaa["forward_return"] > 0
print("passed: feature/target panel is clean and forward-looking only for target")
""",
    "xgb-classifier": """\
panel = pd.DataFrame({
    "date": pd.to_datetime(["2024-01-01"] * 4 + ["2024-01-02"] * 4 + ["2024-01-03"] * 4),
    "ticker": ["A", "B", "C", "D"] * 3,
    "momentum_5d": [0.1, 0.2, -0.1, 0.0, 0.3, 0.2, -0.2, -0.1, 0.2, 0.5, -0.2, -0.3],
    "vol_5d": [0.2, 0.1, 0.4, 0.3, 0.1, 0.2, 0.5, 0.3, 0.2, 0.1, 0.4, 0.5],
    "forward_return": [0.03, 0.04, -0.02, 0.01, 0.05, 0.03, -0.03, -0.01, 0.02, 0.06, -0.04, -0.02],
})
model, preds = fit_tiny_xgb(panel)
assert hasattr(model, "predict_proba")
assert list(preds.columns) == ["date", "ticker", "expected_return"]
assert len(preds) == len(panel)
assert preds["expected_return"].between(0, 1).all()
print("passed: tiny XGBoost model returns probability scores")
""",
    "prediction-contract": """\
raw = pd.DataFrame({
    "date": ["2024-01-01", "2024-01-01"],
    "ticker": ["aapl", "msft"],
    "score": [0.7, 0.4],
})
out = make_prediction_contract(raw, horizon=5)
assert list(out.columns) == ["date", "ticker", "horizon", "expected_return"]
assert out["ticker"].tolist() == ["AAPL", "MSFT"]
assert out["horizon"].tolist() == [5, 5]
assert out["expected_return"].tolist() == [0.7, 0.4]
print("passed: prediction table matches toolkit contract")
""",
    "rank-weights": """\
preds = pd.DataFrame({
    "date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3),
    "ticker": ["A", "B", "C", "A", "B", "C"],
    "expected_return": [0.3, 0.1, 0.2, -0.1, 0.4, 0.2],
})
weights = rank_long_only_weights(preds)
assert weights.index.name == "date"
assert set(weights.columns) == {"A", "B", "C"}
assert abs(float(weights.sum(axis=1).sub(1).abs().max())) < 1e-12
assert weights.loc[pd.Timestamp("2024-01-01"), "A"] > weights.loc[pd.Timestamp("2024-01-01"), "C"] > weights.loc[pd.Timestamp("2024-01-01"), "B"]
assert weights.loc[pd.Timestamp("2024-01-02"), "B"] > weights.loc[pd.Timestamp("2024-01-02"), "C"] > weights.loc[pd.Timestamp("2024-01-02"), "A"]
print("passed: rank long-only weights are date-wise and normalized")
""",
    "minvar-backbone": """\
cov = np.array([[0.04, 0.01, 0.0], [0.01, 0.09, 0.0], [0.0, 0.0, 0.16]])
weights = simple_minvar_weights(cov, ["LOW", "MID", "HIGH"])
assert list(weights.index) == ["LOW", "MID", "HIGH"]
assert abs(float(weights.sum()) - 1.0) < 1e-12
assert (weights >= 0).all()
assert weights["LOW"] > weights["MID"] > weights["HIGH"]
print("passed: simple min-var weights prefer lower variance assets")
""",
    "xgb-tilt": """\
minvar = pd.Series({"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1})
proba = pd.Series({"A": 0.9, "B": 0.2, "C": 0.5, "D": 0.1})
weights = apply_centered_tilt(minvar, proba, tilt_alpha=0.2, max_weight=0.6)
assert abs(float(weights.sum()) - 1.0) < 1e-12
assert (weights >= 0).all() and (weights <= 0.6 + 1e-12).all()
assert weights["A"] > minvar["A"]
assert weights["D"] < minvar["D"]
print("passed: centered XGB signal tilts min-var weights and stays feasible")
""",
    "no-leakage-rebalance": """\
features = pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08"])
assert latest_feature_date_before("2024-01-06", features) == pd.Timestamp("2024-01-05")
assert latest_feature_date_before(pd.Timestamp("2024-01-08"), features) == pd.Timestamp("2024-01-05")
try:
    latest_feature_date_before("2024-01-02", features)
except ValueError:
    print("passed: no prior feature date raises ValueError")
else:
    raise AssertionError("expected ValueError when no prior feature date exists")
""",
    "vol-feature-pack": """\
dates = pd.date_range("2024-01-01", periods=8, freq="D")
prices = pd.DataFrame({
    "date": list(dates) * 2,
    "ticker": ["AAA"] * len(dates) + ["BBB"] * len(dates),
    "adj_close": [10, 11, 10, 12, 11, 13, 12, 14, 20, 20.5, 20.2, 21, 20.6, 21.8, 21.0, 22.0],
})
features = prices.loc[:, ["date", "ticker"]].copy()
iv = pd.DataFrame({"ticker": ["AAA"], "current_iv": [0.42]})
out = add_vol_feature_pack(features, prices, iv, window=3)
assert {"vol_of_vol", "current_iv", "current_iv_missing"}.issubset(out.columns)
assert out.loc[out["ticker"] == "AAA", "current_iv_missing"].eq(0.0).all()
assert out.loc[out["ticker"] == "BBB", "current_iv_missing"].eq(1.0).all()
assert out["current_iv"].notna().all()
print("passed: volatility feature pack adds vol-of-vol and IV missing handling")
""",
    "ew-factor-weights": """\
weights = exponential_sample_weights(5, halflife=2.0)
assert len(weights) == 5
assert np.all(weights > 0)
assert abs(float(weights.mean()) - 1.0) < 1e-12
assert weights[-1] > weights[0]
try:
    exponential_sample_weights(0, halflife=2.0)
except ValueError:
    print("passed: exponentially weighted samples are validated and normalized")
else:
    raise AssertionError("expected ValueError for n=0")
""",
    "factor-covariance": """\
beta = pd.DataFrame(
    [[1.0, 0.5], [0.2, 1.2]],
    index=["AAA", "BBB"],
    columns=["MKT", "SMB"],
)
factor_cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["MKT", "SMB"], columns=["MKT", "SMB"])
resid = pd.Series({"AAA": 0.02, "BBB": 0.03})
out = build_factor_covariance(beta, factor_cov, resid)
expected = beta.to_numpy() @ factor_cov.to_numpy() @ beta.to_numpy().T + np.diag([0.02, 0.03])
assert list(out.index) == ["AAA", "BBB"]
assert list(out.columns) == ["AAA", "BBB"]
assert np.allclose(out.to_numpy(), expected)
assert np.allclose(out.to_numpy(), out.to_numpy().T)
print("passed: factor covariance combines FF covariance and residual risk")
""",
    "submission-manifest": """\
manifest = build_submission_manifest("eamon_xgboost_final", ["vol_20d", "current_iv"], 20, "every_10_trading_days")
assert manifest["model_name"] == "eamon_xgboost_final"
assert manifest["model_family"] == "xgboost"
assert manifest["feature_names"] == ["vol_20d", "current_iv"]
assert manifest["horizon"] == 20
assert manifest["rebalance_frequency"] == "every_10_trading_days"
assert manifest["model_config"]["estimator"] == "XGBClassifier"
assert "build_model_features" in manifest["model_config"]["required_functions"]
assert "predict_from_prices" in manifest["model_config"]["required_functions"]
print("passed: manifest has evaluator-facing metadata")
""",
}


PRELUDE = """\
import math
import numpy as np
import pandas as pd
"""


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    return json.loads(handler.rfile.read(length).decode("utf-8"))


def _lesson_payload() -> dict[str, Any]:
    public_lessons = []
    for lesson in sorted(LESSONS, key=lambda item: int(item["title"].split(".", 1)[0])):
        public_lessons.append({key: value for key, value in lesson.items() if key != "tests"})
    return {
        "title": "Eamon XGBoost Model Dojo",
        "subtitle": "LeetCode-style exercises from first price panel to min-var plus XGBoost tilt.",
        "lessons": public_lessons,
    }


def _run_lesson_code(lesson_id: str, code: str) -> dict[str, Any]:
    if lesson_id not in TESTS:
        raise KeyError(f"unknown lesson id: {lesson_id}")
    script = "\n\n".join([PRELUDE, code, TESTS[lesson_id]])
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as handle:
        handle.write(script)
        path = Path(handle.name)
    try:
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "stdout": exc.stdout or "",
            "stderr": "Timed out after 15 seconds.",
            "returncode": 124,
        }
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    return {
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "returncode": completed.returncode,
    }


class DojoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(STATIC_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/lessons":
            _json_response(self, 200, _lesson_payload())
            return
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            _json_response(self, 404, {"error": "not found"})
            return
        try:
            payload = _read_json(self)
            result = _run_lesson_code(str(payload.get("lesson_id")), str(payload.get("code", "")))
            _json_response(self, 200, result)
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})


def main() -> None:
    port = int(os.environ.get("EAMON_DOJO_PORT", sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PORT))
    server = ThreadingHTTPServer(("127.0.0.1", port), DojoHandler)
    print(f"Eamon XGBoost Model Dojo running at http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
