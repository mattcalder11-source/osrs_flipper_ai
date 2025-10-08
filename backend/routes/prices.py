from fastapi import APIRouter
from osrs_flipper_ai.src.fetch_latest_prices import fetch_latest_prices_dict

router = APIRouter(prefix="/prices", tags=["prices"])

@router.get("/latest")
def get_latest_prices():
    prices = fetch_latest_prices_dict()
    return {"count": len(prices), "prices": prices}
