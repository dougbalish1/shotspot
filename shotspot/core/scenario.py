"""The scenario: where the sensors and the shot are, and what the sensors hear.

Arrival times are computed exactly from geometry:

    tᵢ = t₀ + |p − sᵢ| / c  +  σ·zᵢ

where p is the shot position, t₀ the emission time, sᵢ the sensor position,
and σ·zᵢ optional timing noise. Each sensor keeps a fixed standard-normal draw
zᵢ, so changing σ rescales the same noise instead of reshuffling it. That
makes it easy to watch a solution drift smoothly as noise grows.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

DEFAULT_C = 343.0  # m/s, dry air at about 20 °C
DEFAULT_T0 = 3.25  # s; a clock reading, deliberately not zero
FORMAT_VERSION = 1


@dataclass
class Sensor:
    label: str
    x: float
    y: float
    noise_draw: float = 0.0  # this sensor's fixed N(0, 1) draw; timing error = σ·noise_draw

    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)


@dataclass
class Scenario:
    sensors: list[Sensor] = field(default_factory=list)
    shot: tuple[float, float] | None = None
    t0: float = DEFAULT_T0
    c: float = DEFAULT_C
    sigma: float = 0.0  # timing noise standard deviation, seconds
    reference: str = "S1"  # label of the reference sensor
    width: float = 1000.0
    height: float = 1000.0
    name: str = "Untitled"
    description: str = ""

    # ---- geometry and arrival times -------------------------------------

    def positions(self) -> np.ndarray:
        """Sensor positions as an (N, 2) array."""
        return np.array([s.position for s in self.sensors], dtype=float).reshape(-1, 2)

    def labels(self) -> list[str]:
        return [s.label for s in self.sensors]

    def reference_index(self) -> int:
        """Index of the reference sensor, falling back to the first sensor."""
        labels = self.labels()
        return labels.index(self.reference) if self.reference in labels else 0

    def shot_position(self) -> np.ndarray:
        if self.shot is None:
            raise ValueError("no shot has been placed")
        return np.array(self.shot, dtype=float)

    def distances(self) -> np.ndarray:
        """True distance from the shot to each sensor, |p − sᵢ|."""
        return np.linalg.norm(self.positions() - self.shot_position(), axis=1)

    def true_times(self) -> np.ndarray:
        """Noise-free arrival times, tᵢ = t₀ + |p − sᵢ|/c."""
        return self.t0 + self.distances() / self.c

    def timing_errors(self) -> np.ndarray:
        """The noise added to each sensor's clock reading, σ·zᵢ (seconds)."""
        return self.sigma * np.array([s.noise_draw for s in self.sensors], dtype=float)

    def measured_times(self) -> np.ndarray:
        """What the sensors actually report: true time plus timing noise."""
        return self.true_times() + self.timing_errors()

    # ---- editing ---------------------------------------------------------

    def next_label(self) -> str:
        """The lowest unused label number, so labels (and their colors) stay small."""
        taken = {s.label for s in self.sensors}
        n = 1
        while f"S{n}" in taken:
            n += 1
        return f"S{n}"

    def add_sensor(self, x: float, y: float, rng: np.random.Generator) -> Sensor:
        sensor = Sensor(self.next_label(), float(x), float(y), float(rng.standard_normal()))
        self.sensors.append(sensor)
        if len(self.sensors) == 1:
            self.reference = sensor.label
        return sensor

    def remove_sensor(self, index: int) -> None:
        removed = self.sensors.pop(index)
        if removed.label == self.reference:
            self.reference = self.sensors[0].label if self.sensors else "S1"

    def reroll_noise(self, rng: np.random.Generator) -> None:
        for s in self.sensors:
            s.noise_draw = float(rng.standard_normal())

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "name": self.name,
            "description": self.description,
            "field": {"width": self.width, "height": self.height},
            "speed_of_sound": self.c,
            "t0": self.t0,
            "sigma_s": self.sigma,
            "reference": self.reference,
            "shot": None if self.shot is None else list(self.shot),
            "sensors": [asdict(s) for s in self.sensors],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Scenario:
        return cls(
            sensors=[Sensor(**s) for s in d["sensors"]],
            shot=None if d.get("shot") is None else (float(d["shot"][0]), float(d["shot"][1])),
            t0=d.get("t0", DEFAULT_T0),
            c=d.get("speed_of_sound", DEFAULT_C),
            sigma=d.get("sigma_s", 0.0),
            reference=d.get("reference", "S1"),
            width=d.get("field", {}).get("width", 1000.0),
            height=d.get("field", {}).get("height", 1000.0),
            name=d.get("name", "Untitled"),
            description=d.get("description", ""),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def copy(self) -> Scenario:
        return Scenario.from_dict(self.to_dict())
