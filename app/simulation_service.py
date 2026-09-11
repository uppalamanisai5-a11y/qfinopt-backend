import numpy as np
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Tuple
from app.models import (
    WithdrawalRequest,
    WithdrawalResponse,
    HoldingPeriodReturn,
    ScenarioMetric,
    SipRequest,
    SipResponse,
    SipProjectionRow,
)
from app.data_manager import load_historical_data, match_fund_amfi_nav, get_live_nav_amfi
from app.market_service import get_market_sentiment

def calculate_xirr(amounts: List[float], dates: List[date]) -> float:
    """Calculate XIRR via bisection root-finding."""
    days = np.array([(d - dates[0]).days for d in dates], dtype=float)

    def npv(rate: float) -> float:
        return float(np.sum(np.asarray(amounts) / (1 + rate) ** (days / 365.25)))

    low, high = -0.9999, 2.0
    while npv(high) > 0 and high < 100:
        high *= 2
    for _ in range(120):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
    return (low + high) / 2

def run_withdrawal_simulation(req: WithdrawalRequest) -> WithdrawalResponse:
    """Run Monte Carlo withdrawal timing simulation for selected fund."""
    df = load_historical_data()
    fund_df = df[df["Scheme_Name"] == req.fund_name].sort_values("Date")
    if fund_df.empty:
        raise ValueError(f"Fund '{req.fund_name}' not found in dataset.")

    mu_real = float(fund_df["Daily_Return_%"].mean())
    sigma_real = float(fund_df["Daily_Return_%"].std())
    beta_val = float(fund_df["Beta"].mean())
    risk_level = str(fund_df["Risk_Level"].iloc[-1])

    # Check live nav
    nav_today = get_live_nav_amfi()
    amfi_match = match_fund_amfi_nav(req.fund_name, nav_today)
    nav_source = "📡 LIVE" if amfi_match else "📁 Historical"

    # Market adjustment
    sentiment, month_ret, week_ret = get_market_sentiment()
    try:
        mu_adjusted = mu_real + (week_ret / 500 * beta_val)
    except Exception:
        mu_adjusted = mu_real

    hold_days = 504
    n_sim = req.n_sim
    np.random.seed(42)

    daily_ret = np.random.normal(mu_adjusted / 100.0, sigma_real / 100.0, (hold_days, n_sim))
    paths = np.zeros((hold_days + 1, n_sim))
    paths[0] = req.investment

    for d in range(1, hold_days + 1):
        paths[d] = paths[d - 1] * (1.0 + daily_ret[d - 1])

    exp_v = paths.mean(axis=1)
    prob_g = (paths > req.investment).mean(axis=1)
    vol = paths.std(axis=1)
    risk_adj = (exp_v - req.investment) / (vol + 1.0)
    opt_day = int(np.argmax(risk_adj))
    opt_val = float(exp_v[opt_day])
    gain = opt_val - req.investment
    ret_pct = (gain / req.investment) * 100.0

    # Calculate exact withdrawal date skipping weekends
    invest_dt = datetime.strptime(req.invest_date, "%Y-%m-%d").date()
    current = datetime.combine(invest_dt, datetime.min.time())
    count = 0
    while count < opt_day:
        current += timedelta(days=1)
        if current.weekday() < 5:
            count += 1
    withdraw_dt = current.date()

    # Period returns
    period_returns: List[HoldingPeriodReturn] = []
    for label, days in [("3 months", 63), ("6 months", 126), ("1 year", 252), ("2 years", 504)]:
        d_idx = min(days, hold_days)
        val = float(exp_v[d_idx])
        pct = ((val - req.investment) / req.investment) * 100.0
        p_prob = float(prob_g[d_idx] * 100.0)
        period_returns.append(
            HoldingPeriodReturn(
                label=label,
                days=d_idx,
                expected_value=round(val, 2),
                return_pct=round(pct, 2),
                profit_prob=round(p_prob, 1)
            )
        )

    # Scenarios at optimal exit
    scenarios: List[ScenarioMetric] = []
    opt_slice = paths[opt_day]
    for label, pct_rank in [("Worst 5%", 5), ("Conservative 25%", 25), ("Optimistic 75%", 75), ("Best 95%", 95)]:
        v = float(np.percentile(opt_slice, pct_rank))
        scenarios.append(
            ScenarioMetric(
                label=label,
                percentile=pct_rank,
                value=round(v, 2),
                return_pct=round(((v - req.investment) / req.investment) * 100.0, 2)
            )
        )

    # Downsample curves to ~50 points for mobile charts
    step = max(1, hold_days // 50)
    sample_indices = list(range(0, hold_days + 1, step))
    if hold_days not in sample_indices:
        sample_indices.append(hold_days)
    if opt_day not in sample_indices:
        sample_indices.append(opt_day)
        sample_indices.sort()

    p5_arr = np.percentile(paths, 5, axis=1)
    p95_arr = np.percentile(paths, 95, axis=1)

    chart_days = sample_indices
    chart_expected = [round(float(exp_v[i]), 2) for i in sample_indices]
    chart_p5 = [round(float(p5_arr[i]), 2) for i in sample_indices]
    chart_p95 = [round(float(p95_arr[i]), 2) for i in sample_indices]
    chart_prob_pct = [round(float(prob_g[i] * 100.0), 1) for i in sample_indices]

    return WithdrawalResponse(
        fund_name=req.fund_name,
        invest_date=invest_dt.strftime("%d %b %Y"),
        withdraw_date=withdraw_dt.strftime("%d %b %Y"),
        opt_day=opt_day,
        opt_months=opt_day // 21,
        opt_val=round(opt_val, 2),
        investment=req.investment,
        gain=round(gain, 2),
        ret_pct=round(ret_pct, 2),
        profit_prob=round(float(prob_g[opt_day] * 100.0), 1),
        risk_level=risk_level,
        nav_source=nav_source,
        beta_val=round(beta_val, 3),
        holding_period_returns=period_returns,
        scenario_breakdown=scenarios,
        chart_days=chart_days,
        chart_expected=chart_expected,
        chart_p5=chart_p5,
        chart_p95=chart_p95,
        chart_prob_pct=chart_prob_pct
    )

def run_sip_simulation(req: SipRequest) -> SipResponse:
    """Run lognormal daily return SIP simulation with 21-day deposits and XIRR calculation."""
    df = load_historical_data()
    fund_df = df[df["Scheme_Name"] == req.fund_name].sort_values("Date")
    if fund_df.empty:
        raise ValueError(f"Fund '{req.fund_name}' not found in dataset.")

    # Historical volatility
    nav_log_returns = np.log(fund_df["NAV_Value"] / fund_df["NAV_Value"].shift(1)).dropna()
    historical_vol = float(nav_log_returns.std() * np.sqrt(252) * 100.0) if len(nav_log_returns) > 5 else 18.0
    annual_vol_pct = req.annual_vol_pct if req.annual_vol_pct is not None else float(np.clip(historical_vol, 8.0, 35.0))

    sip_months = req.sip_years * 12
    sip_days_total = req.sip_years * 252
    total_invested = req.sip_amount * sip_months
    n_sip = min(req.n_sim, 5000)

    rng = np.random.default_rng(42)
    annual_return = req.annual_return_pct / 100.0
    annual_vol = annual_vol_pct / 100.0

    daily_log_mean = (np.log1p(annual_return) - 0.5 * annual_vol**2) / 252.0
    daily_log_vol = annual_vol / np.sqrt(252.0)

    daily_returns = np.exp(rng.normal(daily_log_mean, daily_log_vol, size=(sip_days_total, n_sip))) - 1.0

    sip_paths = np.zeros((sip_days_total + 1, n_sip))
    invested_line = np.zeros(sip_days_total + 1)
    portfolio = np.zeros(n_sip)
    deposited = 0.0

    for day in range(sip_days_total):
        if day % 21 == 0:
            portfolio += req.sip_amount
            deposited += req.sip_amount

        portfolio *= (1.0 + daily_returns[day])
        sip_paths[day + 1] = portfolio
        invested_line[day] = deposited
        invested_line[day + 1] = deposited

    final_sip = sip_paths[-1]
    expected_sip = float(final_sip.mean())
    median_sip = float(np.median(final_sip))
    gain_sip = expected_sip - total_invested
    profit_prob = float((final_sip > total_invested).mean() * 100.0)

    # Cashflow dates for XIRR
    invest_dt = datetime.strptime(req.invest_date, "%Y-%m-%d").date()
    cashflow_dates = [invest_dt + timedelta(days=int(30.4375 * m)) for m in range(sip_months)]
    final_dt = invest_dt + timedelta(days=int(365.25 * req.sip_years))

    expected_xirr = calculate_xirr(
        [-req.sip_amount] * sip_months + [expected_sip],
        cashflow_dates + [final_dt]
    ) * 100.0

    # Year-by-year projections
    projections: List[SipProjectionRow] = []
    for yr in range(1, req.sip_years + 1):
        day_index = yr * 252
        y_vals = sip_paths[day_index]
        y_inv = invested_line[day_index]
        projections.append(
            SipProjectionRow(
                year=f"Year {yr}",
                invested=round(float(y_inv), 0),
                expected=round(float(y_vals.mean()), 0),
                median=round(float(np.median(y_vals)), 0),
                range_10_90=f"₹{np.percentile(y_vals, 10):,.0f} - ₹{np.percentile(y_vals, 90):,.0f}",
                profit_probability=round(float((y_vals > y_inv).mean() * 100.0), 1)
            )
        )

    # Downsample curves to ~50 points for mobile charts
    step = max(1, sip_days_total // 50)
    sample_indices = list(range(0, sip_days_total + 1, step))
    if sip_days_total not in sample_indices:
        sample_indices.append(sip_days_total)

    exp_line = sip_paths.mean(axis=1)
    chart_days = sample_indices
    chart_expected = [round(float(exp_line[i]), 0) for i in sample_indices]
    chart_invested = [round(float(invested_line[i]), 0) for i in sample_indices]

    # Final corpus distribution histogram (30 bins)
    counts, bin_edges = np.histogram(final_sip, bins=30)
    distribution_bins = [round(float(b), 0) for b in bin_edges]
    distribution_counts = [int(c) for c in counts]

    return SipResponse(
        monthly_sip=req.sip_amount,
        sip_years=req.sip_years,
        total_invested=round(total_invested, 0),
        expected_sip=round(expected_sip, 0),
        median_sip=round(median_sip, 0),
        gain_sip=round(gain_sip, 0),
        expected_xirr=round(expected_xirr, 1),
        profit_probability=round(profit_prob, 1),
        p10=round(float(np.percentile(final_sip, 10)), 0),
        p90=round(float(np.percentile(final_sip, 90)), 0),
        projections=projections,
        chart_days=chart_days,
        chart_expected=chart_expected,
        chart_invested=chart_invested,
        distribution_bins=distribution_bins,
        distribution_counts=distribution_counts
    )
