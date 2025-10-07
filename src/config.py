import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
BASE_URL = "https://prices.runescape.wiki/api/v1/osrs"
USER_AGENT = os.getenv("USER_AGENT", "OSRS-Flipper/1.0 (contact@example.com)")

# Directories
DATA_DIR = os.getenv("DATA_DIR", "./data")
MODEL_DIR = os.getenv("MODEL_DIR", "./models")

# Schedule settings
SNAPSHOT_INTERVAL_MIN = int(os.getenv("SNAPSHOT_INTERVAL_MIN", 5))
