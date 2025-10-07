# OSRS Flipper AI

An AI-powered Old School RuneScape flipping assistant using the official price API.

## Setup

1. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your environment variables in `.env`:
```
USER_AGENT=OSRS-Flipper/1.0 (contact: you@example.com)
```

3. Run ingestion:
```bash
python src/ingest.py
```

4. Train model:
```bash
python src/train_model.py
```

5. Start API:
```bash
uvicorn src.api:app --reload
```

6. Open dashboard or inspect signals via `/docs`.
