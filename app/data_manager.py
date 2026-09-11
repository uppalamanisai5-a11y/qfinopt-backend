import os
import time
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from app.config import (
    CSV_CACHE_PATH,
    GOOGLE_DRIVE_DOWNLOAD_URL,
    GOOGLE_DRIVE_FALLBACK_URL,
    AMFI_NAV_URL,
    AMFI_NAV_CACHE_TTL,
)

# Global in-memory caches
_historical_df: Optional[pd.DataFrame] = None
_amfi_nav_cache: Dict[str, Dict[str, Any]] = {}
_amfi_cache_time: float = 0.0

def load_historical_data() -> pd.DataFrame:
    """Load historical dataset from disk cache, or download from Google Drive if not cached."""
    global _historical_df
    if _historical_df is not None:
        return _historical_df

    if not CSV_CACHE_PATH.exists():
        print(f"Downloading historical mutual fund dataset to {CSV_CACHE_PATH}...")
        try:
            # First try direct download endpoint
            resp = requests.get(GOOGLE_DRIVE_DOWNLOAD_URL, stream=True, timeout=60)
            if resp.status_code == 200:
                with open(CSV_CACHE_PATH, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            else:
                # Fallback to uc?id=
                resp2 = requests.get(GOOGLE_DRIVE_FALLBACK_URL, stream=True, timeout=60)
                resp2.raise_for_status()
                with open(CSV_CACHE_PATH, "wb") as f:
                    for chunk in resp2.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            print("Download completed successfully.")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            raise RuntimeError(f"Failed to download historical dataset: {e}")

    print("Loading CSV from cache into DataFrame...")
    df = pd.read_csv(CSV_CACHE_PATH)
    df["Date"] = pd.to_datetime(df["Date"])
    _historical_df = df
    print(f"Loaded {len(df):,} records for {df['Scheme_Name'].nunique():,} unique funds.")
    return _historical_df

def get_live_nav_amfi(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """Fetch TODAY's NAV for all funds from AMFI India official API with in-memory TTL caching."""
    global _amfi_nav_cache, _amfi_cache_time
    now = time.time()
    if not force_refresh and _amfi_nav_cache and (now - _amfi_cache_time < AMFI_NAV_CACHE_TTL):
        return _amfi_nav_cache

    try:
        resp = requests.get(AMFI_NAV_URL, timeout=15)
        lines = resp.text.strip().split("\n")
        nav_data = {}
        for line in lines:
            parts = line.strip().split(";")
            if len(parts) >= 6 and parts[0].strip().isdigit():
                try:
                    scheme_code = parts[0].strip()
                    if len(parts) >= 8:
                        base_name = parts[3].strip()
                        plan = parts[4].strip()
                        option = parts[5].strip()
                        scheme_name = f"{base_name} - {plan} - {option}"
                        nav_val = parts[-2].strip()
                        nav_date = parts[-1].strip()
                    else:
                        scheme_name = parts[3].strip()
                        base_name = scheme_name
                        nav_val = parts[4].strip()
                        nav_date = parts[5].strip() if len(parts) > 5 else ""

                    if nav_val not in ["N.A.", "", "#N/A", "-"]:
                        item = {
                            "code": scheme_code,
                            "nav": float(nav_val),
                            "date": nav_date
                        }
                        nav_data[scheme_name] = item
                        if base_name not in nav_data:
                            nav_data[base_name] = item
                except Exception:
                    pass
        if nav_data:
            _amfi_nav_cache = nav_data
            _amfi_cache_time = now
            print(f"AMFI NAV cache refreshed: {len(nav_data):,} funds loaded.")
            return _amfi_nav_cache
    except Exception as e:
        print(f"Failed to fetch AMFI NAV data: {e}")

    return _amfi_nav_cache

def match_fund_amfi_nav(fund_name: str, nav_today: Dict[str, Dict[str, Any]]) -> Optional[Tuple[str, float, str, str]]:
    """
    Fuzzy match fund name against AMFI scheme names using word intersection.
    Returns (matched_name, nav, date, code) if score >= 2 else None.
    """
    if not nav_today or not fund_name:
        return None

    if fund_name in nav_today:
        val = nav_today[fund_name]
        return fund_name, val["nav"], val["date"], val["code"]

    best_match = None
    best_score = 0
    fn_lower = fund_name.lower()
    fund_words = set(fn_lower.replace("-", " ").replace("–", " ").replace("'", " ").split())

    for amfi_name in nav_today:
        an_lower = amfi_name.lower()
        amfi_words = set(an_lower.replace("-", " ").replace("–", " ").replace("'", " ").split())
        score = len(fund_words & amfi_words)
        if "direct" in fn_lower and "direct" in an_lower:
            score += 2
        if "growth" in fn_lower and "growth" in an_lower:
            score += 1
        if score > best_score:
            best_score = score
            best_match = amfi_name

    if best_match and best_score >= 2:
        val = nav_today[best_match]
        return best_match, val["nav"], val["date"], val["code"]
    return None

def get_categories_and_risk_levels() -> Tuple[List[str], List[str]]:
    """Return sorted unique categories and risk levels from dataset."""
    df = load_historical_data()
    categories = sorted(df["Sheet_Category"].dropna().unique().tolist())
    risk_levels = sorted(df["Risk_Level"].dropna().unique().tolist())
    return categories, risk_levels

def filter_funds(categories: Optional[List[str]] = None, risk_levels: Optional[List[str]] = None) -> List[str]:
    """Filter fund names by selected categories and risk levels."""
    df = load_historical_data()
    sub_df = df
    if categories:
        sub_df = sub_df[sub_df["Sheet_Category"].isin(categories)]
    if risk_levels:
        sub_df = sub_df[sub_df["Risk_Level"].isin(risk_levels)]

    return sorted(sub_df["Scheme_Name"].dropna().unique().tolist())
