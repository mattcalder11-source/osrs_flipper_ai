from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routes import flips, prices

app = FastAPI(title="OSRS AI Flipper API", version="1.0")

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(flips.router)
app.include_router(prices.router)

@app.get("/")
def root():
    return {"status": "ok", "message": "OSRS Flipper backend is running!"}
