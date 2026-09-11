import os
from pathlib import Path
from datetime import timezone, timedelta

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CSV_CACHE_PATH = DATA_DIR / "qfinopt_cleaned.csv"
GOOGLE_DRIVE_FILE_ID = "1EfNv54tjvjcsJdZhxYwa4zPJxl9q9_9X"
GOOGLE_DRIVE_DOWNLOAD_URL = f"https://drive.usercontent.google.com/download?id={GOOGLE_DRIVE_FILE_ID}"
GOOGLE_DRIVE_FALLBACK_URL = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
MFAPI_SEARCH_URL = "https://api.mfapi.in/mf/search"
MFAPI_HIST_URL = "https://api.mfapi.in/mf"

# Timezones
IST = timezone(timedelta(hours=5, minutes=30))

# Cache TTLs (seconds)
MARKET_CACHE_TTL = 300  # 5 minutes
AMFI_NAV_CACHE_TTL = 3600  # 1 hour
HISTORICAL_DATA_CACHE_TTL = 86400  # 24 hours
MFAPI_SEARCH_CACHE_TTL = 86400  # 24 hours
MFAPI_HIST_CACHE_TTL = 3600  # 1 hour
