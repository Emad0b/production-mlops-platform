from pydantic import BaseModel, Field, ConfigDict


class Reading(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    temperature: float = Field(ge=0, le=160)
    vibration: float = Field(ge=0, le=30)
    pressure: float = Field(ge=0, le=20)
    rpm: float = Field(ge=100, le=10000)
    operating_hours: float = Field(ge=0, le=100000)


class Prepare(BaseModel):
    rows: int = Field(default=4000, ge=100, le=100000)
    seed: int = Field(default=42, ge=0, le=2147483647)
    shifted: bool = False


class DatasetChoice(BaseModel):
    dataset_id: str


class ModelChoice(BaseModel):
    model_id: str


class MonitorRequest(BaseModel):
    reference_id: str
    current_id: str


class Label(BaseModel):
    prediction_id: str
    failure: int = Field(ge=0, le=1)
