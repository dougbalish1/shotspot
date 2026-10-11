import numpy as np

C = 343.0
T0 = 3.25


def arrival_times(sensors, shot, t0=T0, c=C):
    """Exact arrival times, computed independently of Scenario."""
    return t0 + np.linalg.norm(np.asarray(sensors, float) - np.asarray(shot, float), axis=1) / c
