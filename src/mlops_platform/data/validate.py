import pandera.pandas as pa
from ..config import FEATURES

RANGES = {
    "temperature": (0, 160),
    "vibration": (0, 30),
    "pressure": (0, 20),
    "rpm": (100, 10000),
    "operating_hours": (0, 100000),
}
schema = pa.DataFrameSchema(
    {
        "record_id": pa.Column(str, unique=True),
        "timestamp": pa.Column(float, pa.Check.ge(0)),
        **{
            name: pa.Column(float, pa.Check.in_range(*RANGES[name]))
            for name in FEATURES
        },
        "failure": pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
    coerce=True,
)


def validate(frame):
    result = schema.validate(frame, lazy=True)
    if len(result) < 100:
        raise ValueError("At least 100 records are required")
    if result["timestamp"].duplicated().any():
        raise ValueError("Timestamps must be unique")
    if result["failure"].nunique() != 2:
        raise ValueError("Both failure classes are required")
    return result
