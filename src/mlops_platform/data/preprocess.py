from ..config import FEATURES


def temporal_split(frame):
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    train = frame.iloc[: int(len(frame) * 0.6)]
    validation = frame.iloc[int(len(frame) * 0.6) : int(len(frame) * 0.8)]
    test = frame.iloc[int(len(frame) * 0.8) :]
    sets = [set(s.record_id) for s in (train, validation, test)]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Record leakage between splits")
    if not train.timestamp.max() < validation.timestamp.min() < test.timestamp.min():
        raise ValueError("Temporal boundaries overlap")
    if any(s.failure.nunique() != 2 for s in (train, validation, test)):
        raise ValueError("Each temporal split must contain both classes")
    return train, validation, test
