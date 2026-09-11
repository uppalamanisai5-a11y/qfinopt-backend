import time
from datetime import datetime
from typing import Dict, Any, Tuple
from app.config import IST, MARKET_CACHE_TTL
from app.models import MarketIndex, MarketResponse

_market_cache: Dict[str, Any] = {}
_market_cache_time: float = 0.0

def fetch_live_market(force_refresh: bool = False) -> MarketResponse:
    """Fetch live Nifty + Sensex from Yahoo Finance with 5 minute TTL caching."""
    global _market_cache, _market_cache_time
    now = time.time()
    if not force_refresh and _market_cache and (now - _market_cache_time < MARKET_CACHE_TTL):
        return _market_cache["data"]

    import yfinance as yf

    result_indices: Dict[str, MarketIndex] = {}
    indices = [
        ("Nifty 50",   "^NSEI"),
        ("Sensex",     "^BSESN"),
        ("Bank Nifty", "^NSEBANK"),
        ("Nifty IT",   "^CNXIT")
    ]

    for name, ticker in indices:
        try:
            tk = yf.Ticker(ticker)
            price_found = False
            # Method 1: fast_info
            try:
                fi = tk.fast_info
                now_val = float(fi.last_price)
                prev_val = float(fi.previous_close)
                if now_val > 0 and prev_val > 0:
                    chg = now_val - prev_val
                    pct = (chg / prev_val) * 100
                    result_indices[name] = MarketIndex(
                        name=name, price=round(now_val, 2), change=round(chg, 2), pct=round(pct, 2)
                    )
                    price_found = True
            except Exception:
                pass

            # Method 2: history
            if not price_found:
                try:
                    hist = tk.history(period="5d", interval="1d")
                    if len(hist) >= 2:
                        now_val = float(hist["Close"].iloc[-1])
                        prev_val = float(hist["Close"].iloc[-2])
                        chg = now_val - prev_val
                        pct = (chg / prev_val) * 100
                        result_indices[name] = MarketIndex(
                            name=name, price=round(now_val, 2), change=round(chg, 2), pct=round(pct, 2)
                        )
                except Exception:
                    pass
        except Exception:
            pass

    # Sentiment analysis from Nifty
    month_ret, week_ret, sentiment = 0.0, 0.0, "Neutral"
    try:
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1mo")
        if len(hist) > 0:
            month_ret = (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[0]) - 1) * 100
            week_ret = (float(hist["Close"].iloc[-1]) / float(hist["Close"].iloc[-5]) - 1) * 100 if len(hist) >= 5 else 0.0
            sentiment = "Bullish" if month_ret > 2.0 else "Bearish" if month_ret < -2.0 else "Neutral"
    except Exception:
        pass

    last_updated_str = datetime.now(IST).strftime("%d %b %Y %I:%M %p") + " IST"

    response = MarketResponse(
        indices=result_indices,
        sentiment=sentiment,
        month_return=round(month_ret, 2),
        week_return=round(week_ret, 2),
        last_updated=last_updated_str
    )

    _market_cache = {"data": response}
    _market_cache_time = now
    return response

def get_market_sentiment() -> Tuple[str, float, float]:
    """Return current sentiment, month return, week return."""
    market_resp = fetch_live_market()
    return market_resp.sentiment, market_resp.month_return, market_resp.week_return
