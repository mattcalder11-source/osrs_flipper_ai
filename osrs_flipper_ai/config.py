"""
Global configuration for OSRS Flipper AI.
Loads values from .env file when available.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Core settings ---
BASE_URL = os.getenv("BASE_URL", "https://prices.runescape.wiki/api/v1/osrs")
USER_AGENT = os.getenv("USER_AGENT", "osrs_flipper_ai/1.0 (default)")
DATA_DIR = os.getenv("DATA_DIR", str(Path("data")))
TARGET_COL = "net_profit_pct"

# Ensure base data dir exists
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
