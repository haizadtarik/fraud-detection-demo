from pydantic import BaseModel, Field
from typing import Literal

class Transaction(BaseModel):
    transaction_type: Literal["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
    amount: float = Field(gt=0, description="Transaction amount")
    name_orig: str = Field(description="Originating account id")
    name_dest: str = Field(description="Destination account id")
    step: int = Field(ge=0, description="Simulated hour-of-sim; real systems pass a timestamp")

class ScoreResponse(BaseModel):
    decision: Literal["ALLOW", "STEP_UP_AUTH", "HOLD", "BLOCK"]
    fraud_probability: float
    reason: str
    rule_triggered: str | None
    latency_ms: float