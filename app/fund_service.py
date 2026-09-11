import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from app.config import MFAPI_SEARCH_URL, MFAPI_HIST_URL, IST
from app.models import (
    FundStatsResponse,
    FundHistoryResponse,
    CompareRequest,
    CompareResponse,
    CompareSeries,
    CompareFundStats,
    PlatformRecommendationResponse,
    PlatformItem,
)
from app.data_manager import (
    load_historical_data,
    get_live_nav_amfi,
    match_fund_amfi_nav,
)
from app.market_service import get_market_sentiment

# Cache for resolved mfapi codes
_mfapi_code_cache: Dict[str, str] = {}
# Cache for mfapi history
_mfapi_hist_cache: Dict[str, pd.DataFrame] = {}

def get_fund_stats(fund_name: str) -> FundStatsResponse:
    """Retrieve full statistical indicators for a mutual fund."""
    df = load_historical_data()
    fund_df = df[df["Scheme_Name"] == fund_name].sort_values("Date")
    if fund_df.empty:
        raise ValueError(f"Fund '{fund_name}' not found.")

    mu_real = float(fund_df["Daily_Return_%"].mean())
    sigma_real = float(fund_df["Daily_Return_%"].std())
    sharpe_val = float(fund_df["Sharpe"].mean())
    alpha_val = float(fund_df["Alpha"].mean())
    beta_val = float(fund_df["Beta"].mean())
    expense = float(fund_df["Expense_Ratio"].mean())
    risk_level = str(fund_df["Risk_Level"].iloc[-1])
    category = str(fund_df["Sheet_Category"].iloc[-1])
    hist_nav = float(fund_df["NAV_Value"].iloc[-1])

    nav_1y = float(fund_df["NAV_Value"].iloc[-252] if len(fund_df) > 252 else fund_df["NAV_Value"].iloc[0])
    ret_1y = ((hist_nav - nav_1y) / nav_1y) * 100.0

    # AMFI Live NAV check
    nav_today = get_live_nav_amfi()
    amfi_match = match_fund_amfi_nav(fund_name, nav_today)
    if amfi_match:
        display_nav = amfi_match[1]
        nav_source = "📡 LIVE"
    else:
        display_nav = hist_nav
        nav_source = "📁 Historical"

    sentiment, month_ret, week_ret = get_market_sentiment()
    try:
        mu_adjusted = mu_real + (week_ret / 500.0 * beta_val)
    except Exception:
        mu_adjusted = mu_real

    return FundStatsResponse(
        fund_name=fund_name,
        category=category,
        risk_level=risk_level,
        display_nav=round(display_nav, 2),
        nav_source=nav_source,
        ret_1y=round(ret_1y, 2),
        sharpe_val=round(sharpe_val, 3),
        alpha_val=round(alpha_val, 3),
        beta_val=round(beta_val, 3),
        expense=round(expense, 2),
        mu_real=round(mu_real, 4),
        sigma_real=round(sigma_real, 4),
        mu_adjusted=round(mu_adjusted, 4),
        sentiment=sentiment
    )

def resolve_mfapi_code(fund_name: str, fallback_code: str) -> str:
    """Resolve fund name to mfapi.in scheme code."""
    if fund_name in _mfapi_code_cache:
        return _mfapi_code_cache[fund_name]

    # Method 1: Check live AMFI cache first (AMFI codes map 1:1 to mfapi.in codes)
    try:
        nav_today = get_live_nav_amfi()
        amfi_match = match_fund_amfi_nav(fund_name, nav_today)
        if amfi_match and amfi_match[3]:
            resolved = str(amfi_match[3])
            _mfapi_code_cache[fund_name] = resolved
            return resolved
    except Exception:
        pass

    # Method 2: Search mfapi.in API
    try:
        resp = requests.get(MFAPI_SEARCH_URL, params={"q": fund_name}, timeout=10)
        matches = resp.json() if resp.status_code == 200 else []

        if not matches:
            clean_words = [
                w for w in fund_name.replace("-", " ").replace("–", " ").replace("'", " ").replace("’", " ").split()
                if w.lower() not in ["fund", "scheme", "mutual", "direct", "growth", "plan"]
            ]
            if clean_words:
                short_q = " ".join(clean_words[:3])
                resp2 = requests.get(MFAPI_SEARCH_URL, params={"q": short_q}, timeout=10)
                if resp2.status_code == 200:
                    matches = resp2.json()

        if matches:
            fn_lower = fund_name.lower()
            def score(m):
                n = m.get("schemeName", "").lower()
                s = 0
                if "direct" in fn_lower and "direct" in n: s += 3
                elif "direct" not in fn_lower and "direct" not in n: s += 1
                if "growth" in fn_lower and "growth" in n: s += 2
                return s

            best = max(matches, key=score)
            resolved = str(best.get("schemeCode", fallback_code))
            _mfapi_code_cache[fund_name] = resolved
            return resolved
    except Exception:
        pass

    _mfapi_code_cache[fund_name] = fallback_code
    return fallback_code

def get_live_nav_history_mfapi(code: str) -> Optional[pd.DataFrame]:
    """Fetch daily NAV history for a fund from mfapi.in."""
    if code in _mfapi_hist_cache:
        return _mfapi_hist_cache[code]

    try:
        resp = requests.get(f"{MFAPI_HIST_URL}/{code}", timeout=10)
        if resp.status_code == 200:
            payload = resp.json()
            hist = payload.get("data", [])
            if hist:
                hdf = pd.DataFrame(hist)
                hdf["date"] = pd.to_datetime(hdf["date"], dayfirst=True, errors="coerce")
                hdf["nav"] = pd.to_numeric(hdf["nav"], errors="coerce")
                hdf = hdf.dropna().sort_values("date").reset_index(drop=True)
                if len(hdf) > 5:
                    _mfapi_hist_cache[code] = hdf
                    return hdf
    except Exception:
        pass

    return None

def get_fund_history(
    fund_name: str,
    period: str = "1Y",
    show_forecast: bool = True,
    forecast_days: int = 30
) -> FundHistoryResponse:
    """Get historical percentage change chart data and trend forecast."""
    df = load_historical_data()
    fund_df = df[df["Scheme_Name"] == fund_name].sort_values("Date")
    if fund_df.empty:
        raise ValueError(f"Fund '{fund_name}' not found.")

    fallback_code = str(fund_df["Scheme_Code"].iloc[-1]).split(".")[0]
    scheme_code = resolve_mfapi_code(fund_name, fallback_code)

    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "5Y": 1260, "ALL": 100000}
    n_days = period_map.get(period, 252)

    live_hist = get_live_nav_history_mfapi(scheme_code)
    if live_hist is not None and len(live_hist) > 5:
        plot_df = live_hist.tail(n_days) if n_days < len(live_hist) else live_hist
        base = float(plot_df["nav"].iloc[0])
        dates = [d.strftime("%d %b %Y") for d in plot_df["date"]]
        raw_dates = list(plot_df["date"])
        values = [round(float((v / base - 1.0) * 100.0), 2) for v in plot_df["nav"]]
        source_label = "📡 Live daily NAV history - mfapi.in (AMFI data)"
    else:
        plot_df = fund_df.tail(n_days) if n_days < len(fund_df) else fund_df
        base = float(plot_df["NAV_Value"].iloc[0])
        dates = [d.strftime("%d %b %Y") for d in plot_df["Date"]]
        raw_dates = list(plot_df["Date"])
        values = [round(float((v / base - 1.0) * 100.0), 2) for v in plot_df["NAV_Value"]]
        source_label = "📁 Historical CSV dataset"

    # Calculate recent trend
    recent_window = values[-30:] if len(values) > 30 else values
    recent_rets = np.diff(recent_window) if len(recent_window) > 1 else [0]
    mu_trend = float(np.mean(recent_rets)) if len(recent_rets) > 0 else 0.0

    if mu_trend > 0.01:
        trend_label, trend_icon = "Trending UP", "📈"
    elif mu_trend < -0.01:
        trend_label, trend_icon = "Trending DOWN", "📉"
    else:
        trend_label, trend_icon = "Flat / Sideways", "➖"

    # Forecast
    fut_dates: List[str] = []
    fut_vals: List[float] = []
    predicted_final: Optional[float] = None
    predicted_nav: Optional[float] = None

    if show_forecast and values and raw_dates:
        last_val = values[-1]
        last_date = raw_dates[-1]
        forecast_b_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=forecast_days)
        fut_dates = [d.strftime("%d %b %Y") for d in forecast_b_dates]
        fut_vals = [round(float(last_val + mu_trend * (i + 1)), 2) for i in range(forecast_days)]
        if fut_vals:
            predicted_final = fut_vals[-1]
            predicted_nav = round(base * (1.0 + predicted_final / 100.0), 2)

    # Histogram of daily returns
    rets = fund_df["Daily_Return_%"].dropna()
    counts, bin_edges = np.histogram(rets, bins=25)
    hist_bins = [round(float(b), 2) for b in bin_edges]
    hist_counts = [int(c) for c in counts]

    # Stats table rows
    mu_real = float(fund_df["Daily_Return_%"].mean())
    sigma_real = float(fund_df["Daily_Return_%"].std())
    sharpe = float(fund_df["Sharpe"].mean())
    alpha = float(fund_df["Alpha"].mean())
    beta = float(fund_df["Beta"].mean())
    expense = float(fund_df["Expense_Ratio"].mean())
    risk = str(fund_df["Risk_Level"].iloc[-1])
    hist_nav = float(fund_df["NAV_Value"].iloc[-1])
    nav_1y = float(fund_df["NAV_Value"].iloc[-252] if len(fund_df) > 252 else fund_df["NAV_Value"].iloc[0])
    ret_1y = ((hist_nav - nav_1y) / nav_1y) * 100.0

    stats_list = [
        {"metric": "Daily Return (avg)", "value": f"{mu_real:.4f}%"},
        {"metric": "Annual Return (est)", "value": f"{mu_real * 252:.2f}%"},
        {"metric": "Volatility (SD)", "value": f"{sigma_real:.4f}%"},
        {"metric": "Sharpe Ratio", "value": f"{sharpe:.3f}"},
        {"metric": "Alpha", "value": f"{alpha:.3f}"},
        {"metric": "Beta", "value": f"{beta:.3f}"},
        {"metric": "Expense Ratio", "value": f"{expense:.2f}%"},
        {"metric": "Risk Level", "value": risk},
        {"metric": "Historical NAV", "value": f"₹{hist_nav:.2f}"},
        {"metric": "1 Year Return", "value": f"{ret_1y:.2f}%"}
    ]

    return FundHistoryResponse(
        fund_name=fund_name,
        scheme_code=scheme_code,
        period=period,
        source_label=source_label,
        trend_label=trend_label,
        trend_icon=trend_icon,
        current_pct=values[-1] if values else 0.0,
        predicted_pct=predicted_final,
        predicted_nav=predicted_nav,
        actual_dates=dates,
        actual_pct=values,
        forecast_dates=fut_dates,
        forecast_pct=fut_vals,
        hist_return_bins=hist_bins,
        hist_return_counts=hist_counts,
        stats=stats_list
    )

def compare_funds(req: CompareRequest) -> CompareResponse:
    """Compare multiple mutual funds with % change curves and stats table."""
    df = load_historical_data()
    period_map = {"1M": 21, "3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "3Y": 756, "ALL": 100000}
    n_days = period_map.get(req.period, 252)

    tab_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

    series_list: List[CompareSeries] = []
    table_list: List[CompareFundStats] = []
    nav_today = get_live_nav_amfi()

    for idx, fund in enumerate(req.fund_names[:15]):
        full_fdf = df[df["Scheme_Name"] == fund].sort_values("Date")
        if full_fdf.empty or len(full_fdf) < 5:
            continue

        fdf = full_fdf.tail(n_days) if n_days < len(full_fdf) else full_fdf
        base_nav = float(fdf["NAV_Value"].iloc[0])
        dates = [d.strftime("%d %b %Y") for d in fdf["Date"]]
        raw_dates = list(fdf["Date"])
        values = [round(float((v / base_nav - 1.0) * 100.0), 2) for v in fdf["NAV_Value"]]

        # Check live AMFI NAV continuation
        amfi_match = match_fund_amfi_nav(fund, nav_today)
        if amfi_match:
            live_val = round(float((amfi_match[1] / base_nav - 1.0) * 100.0), 2)
            if abs(live_val - values[-1]) <= max(15.0, abs(values[-1]) * 0.5):
                dates.append("Today")
                values.append(live_val)

        # Forecast
        fut_dates: List[str] = []
        fut_vals: List[float] = []
        if req.show_forecast and raw_dates:
            rets = fdf["Daily_Return_%"].dropna().tail(90)
            mu = float(rets.mean()) if len(rets) > 0 else 0.0
            last_val = values[-1]
            last_date = raw_dates[-1]
            b_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=req.forecast_days)
            fut_dates = [d.strftime("%d %b %Y") for d in b_dates]
            fut_vals = [round(float(last_val + mu * (i + 1)), 2) for i in range(req.forecast_days)]

        series_list.append(
            CompareSeries(
                fund_name=fund,
                color=tab_colors[idx % len(tab_colors)],
                actual_dates=dates,
                actual_pct=values,
                forecast_dates=fut_dates,
                forecast_pct=fut_vals
            )
        )

        hist_nav = float(full_fdf["NAV_Value"].iloc[-1])
        nav_1y = float(full_fdf["NAV_Value"].iloc[-252] if len(full_fdf) > 252 else full_fdf["NAV_Value"].iloc[0])
        r1y = ((hist_nav - nav_1y) / nav_1y) * 100.0

        table_list.append(
            CompareFundStats(
                fund_name=fund,
                sharpe=round(float(full_fdf["Sharpe"].mean()), 3),
                ret_1y=round(r1y, 2),
                alpha=round(float(full_fdf["Alpha"].mean()), 3),
                beta=round(float(full_fdf["Beta"].mean()), 3),
                expense=round(float(full_fdf["Expense_Ratio"].mean()), 3),
                risk=str(full_fdf["Risk_Level"].iloc[-1])
            )
        )

    return CompareResponse(period=req.period, series=series_list, table=table_list)

def get_platform_recommendation(investment: float) -> PlatformRecommendationResponse:
    """Generate platform recommendation and personalised action plan based on investment amount."""
    if investment < 10000:
        top = "Groww"
        reason = "Best for small amounts · zero minimum"
    elif investment < 100000:
        top = "Kuvera"
        reason = "Best free direct fund platform"
    else:
        top = "Zerodha Coin"
        reason = "Best analytics for large investments"

    platforms = [
        PlatformItem(name="Groww", url="https://groww.in", rating="⭐⭐⭐⭐⭐", desc="Zero commission · ₹100 min · Instant KYC"),
        PlatformItem(name="Zerodha Coin", url="https://coin.zerodha.com", rating="⭐⭐⭐⭐⭐", desc="Best analytics · Direct funds · Stocks + MF"),
        PlatformItem(name="Kuvera", url="https://kuvera.in", rating="⭐⭐⭐⭐⭐", desc="100% free · Tax harvesting · Goal planning"),
        PlatformItem(name="Paytm Money", url="https://paytmmoney.com", rating="⭐⭐⭐⭐", desc="UPI instant · SIP automation"),
        PlatformItem(name="MF Central", url="https://mfcentral.com", rating="⭐⭐⭐⭐", desc="SEBI official · Most secure · Free"),
    ]

    action_steps = [
        f"Step 1: Open {top} and complete your 5-minute KYC with Aadhaar & PAN.",
        "Step 2: Search for your selected mutual fund inside the platform.",
        f"Step 3: Set up your investment (₹{investment:,.0f} Lump Sum or SIP).",
        "Step 4: Set a calendar or WhatsApp reminder for your calculated optimal withdrawal date.",
        "Step 5: Review portfolio performance semi-annually."
    ]

    return PlatformRecommendationResponse(
        investment_amount=investment,
        top_platform=top,
        reason=reason,
        platforms=platforms,
        action_plan_steps=action_steps
    )
