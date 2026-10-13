import numpy as np
from mlops_platform.monitoring.drift import psi


def test_psi_uses_underflow_and_overflow_bins():
    reference = np.linspace(0, 1, 1000)
    assert psi(reference, reference) < 1e-9
    assert psi(reference, reference + 2) > 0.2
