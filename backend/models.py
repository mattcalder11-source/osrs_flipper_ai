from sqlmodel import SQLModel, Field, create_engine
from datetime import datetime

class Flip(SQLModel, table=True):
    item_id: int = Field(primary_key=True)
    name: str
    entry_price: float
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    spread_ratio: float = 0.0
    rsi: float = 50.0
    momentum: float = 0.0

engine = create_engine("sqlite:///../data/flips.db")
SQLModel.metadata.create_all(engine)
