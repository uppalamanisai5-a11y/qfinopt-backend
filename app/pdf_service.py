import io
from datetime import datetime
from fpdf import FPDF
from app.config import IST
from app.models import PdfReportRequest, WithdrawalRequest, SipRequest
from app.fund_service import get_fund_stats
from app.simulation_service import run_withdrawal_simulation, run_sip_simulation

def generate_pdf_report(req: PdfReportRequest) -> bytes:
    """Generate a clean, professional PDF investment report using fpdf2."""
    stats = get_fund_stats(req.fund_name)

    withdraw_req = WithdrawalRequest(
        fund_name=req.fund_name,
        invest_date=req.invest_date,
        investment=req.investment,
        n_sim=2000
    )
    w_res = run_withdrawal_simulation(withdraw_req)

    sip_req = SipRequest(
        fund_name=req.fund_name,
        invest_date=req.invest_date,
        sip_amount=req.sip_amount,
        sip_years=req.sip_years,
        n_sim=2000
    )
    s_res = run_sip_simulation(sip_req)

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(190, 12, "Q-FinOpt Investment Report", 0, 1, "C")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(190, 8, f"For: {req.user_name} | Date: {datetime.now(IST).strftime('%d %b %Y %I:%M %p')} IST", 0, 1, "C")
    pdf.ln(5)

    # Fund Details Section
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 10, "Fund Details", 0, 1)
    pdf.set_font("Helvetica", "", 10)
    for label, value in [
        ("Fund Name", req.fund_name[:60]),
        ("Category", stats.category),
        ("Risk Level", stats.risk_level),
        ("Latest NAV", f"Rs {stats.display_nav:.2f} ({stats.nav_source})"),
        ("1 Year Return", f"{stats.ret_1y:.2f}%"),
        ("Sharpe Ratio", f"{stats.sharpe_val:.3f}"),
        ("Alpha", f"{stats.alpha_val:.3f}"),
        ("Beta", f"{stats.beta_val:.3f}"),
        ("Expense Ratio", f"{stats.expense:.2f}%"),
        ("Market Sentiment", stats.sentiment),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, label + ":", 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 7, str(value), 0, 1)
    pdf.ln(4)

    # Withdrawal Recommendation
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 10, "Withdrawal Recommendation", 0, 1)
    for label, value in [
        ("Investment Amount", f"Rs {req.investment:,.0f}"),
        ("Investment Date", w_res.invest_date),
        ("Withdraw On", w_res.withdraw_date),
        ("Holding Period", f"{w_res.opt_day} days (~{w_res.opt_months} months)"),
        ("Expected Value", f"Rs {w_res.opt_val:,.0f}"),
        ("Expected Gain", f"Rs {w_res.gain:,.0f} ({w_res.ret_pct:.1f}%)"),
        ("Profit Probability", f"{w_res.profit_prob:.1f}%"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, label + ":", 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 7, str(value), 0, 1)
    pdf.ln(4)

    # SIP Projection
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(190, 10, "SIP Projection", 0, 1)
    for label, value in [
        ("Monthly SIP", f"Rs {req.sip_amount:,.0f}"),
        ("Duration", f"{req.sip_years} years"),
        ("Total Invested", f"Rs {s_res.total_invested:,.0f}"),
        ("Expected Corpus", f"Rs {s_res.expected_sip:,.0f}"),
        ("Wealth Gain", f"Rs {s_res.gain_sip:,.0f}"),
        ("Est. XIRR", f"{s_res.expected_xirr:.1f}% per year"),
    ]:
        pdf.set_font("Helvetica", "B", 10)
        pdf.cell(70, 7, label + ":", 0, 0)
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(120, 7, str(value), 0, 1)
    pdf.ln(4)

    # Disclaimer
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.multi_cell(
        190, 5,
        "DISCLAIMER: For educational purposes only. "
        "Mutual fund investments are subject to market risks. "
        "Past performance does not guarantee future returns."
    )

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_xy(10, 280)
    pdf.cell(
        190, 6,
        f"Q-FinOpt Mobile Edition | Built by MANI SAI | {datetime.now(IST).strftime('%d %b %Y')}",
        0, 0, "C"
    )

    return bytes(pdf.output())
