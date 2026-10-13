import numpy as np
import pytest
from pandera.errors import SchemaErrors
from mlops_platform.data.ingest import simulate, save_dataset, load_dataset
from mlops_platform.data.validate import validate
from mlops_platform.data.preprocess import temporal_split


def test_reproducible_checksum_and_leakage_checks():
    frame = simulate(300, 1)
    first = save_dataset(frame, {"seed": 1})
    second = save_dataset(simulate(300, 1), {"seed": 1})
    assert first["sha256"] == second["sha256"]
    assert len(load_dataset(first["id"])) == 300
    train, validation, test = temporal_split(frame)
    assert train.timestamp.max() < validation.timestamp.min() < test.timestamp.min()
    assert set(train.record_id).isdisjoint(test.record_id)


def test_invalid_ranges_missing_features_duplicates_and_nan():
    frame = simulate(300, 3)
    invalid = frame.copy()
    invalid.loc[0, "temperature"] = 999
    with pytest.raises(SchemaErrors):
        validate(invalid)
    with pytest.raises(SchemaErrors):
        validate(frame.drop(columns=["rpm"]))
    invalid = frame.copy()
    invalid.loc[0, "record_id"] = invalid.loc[1, "record_id"]
    with pytest.raises(SchemaErrors):
        validate(invalid)
    invalid = frame.copy()
    invalid.loc[0, "vibration"] = np.nan
    with pytest.raises(SchemaErrors):
        validate(invalid)
