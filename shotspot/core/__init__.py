"""Pure-numpy model and solvers. Nothing in this package may import Qt.

Conventions used throughout:
    positions   meters, numpy arrays of shape (2,) or (N, 2)
    times       seconds, absolute clock readings (not relative to the shot)
    c           speed of sound, m/s
    reference   index of the reference sensor ("sensor k" in the derivations)
"""
