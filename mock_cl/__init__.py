"""
Mock CL (Cortical Labs) API for offline development.

Simulates the biological neural network on the CL1's 8x8 MEA (Multi-Electrode
Array) so you can develop reservoir computing pipelines without hardware.

When your mentor connects you to a real CL1, you swap `import mock_cl as cl`
for `import cl` and everything else stays the same.
"""

import numpy as np
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import List, Optional

# ── MEA layout constants ────────────────────────────────────────────────────
FRAMES_PER_SECOND = 25_000
TOTAL_CHANNELS = 64
UNUSED_CHANNELS = {0, 7, 56, 63}
REFERENCE_CHANNEL = 4
ACTIVE_CHANNELS = sorted(
    set(range(TOTAL_CHANNELS)) - UNUSED_CHANNELS - {REFERENCE_CHANNEL}
)  # 59 usable electrodes


# ── API data classes (mirror the real cl module) ─────────────────────────────

@dataclass
class Spike:
    channel: int
    timestamp: int
    _samples: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def samples(self) -> np.ndarray:
        if self._samples is None:
            # Synthetic spike waveform: 75 samples (25 pre + 50 post)
            t = np.arange(75)
            self._samples = (
                -80.0 * np.exp(-((t - 25) ** 2) / 18)  # negative peak
                + 30.0 * np.exp(-((t - 35) ** 2) / 50)  # positive rebound
                + np.random.randn(75) * 3.0              # noise
            )
        return self._samples


@dataclass
class TickAnalysis:
    spikes: List[Spike] = field(default_factory=list)


@dataclass
class Tick:
    iteration: int
    timestamp: int
    analysis: TickAnalysis = field(default_factory=TickAnalysis)


class ChannelSet:
    def __init__(self, *channels):
        flat = []
        for c in channels:
            if isinstance(c, (list, tuple, np.ndarray)):
                flat.extend(int(x) for x in c)
            else:
                flat.append(int(c))
        self.channels = sorted(set(flat))

    def __iter__(self):
        return iter(self.channels)

    def __repr__(self):
        return f"ChannelSet({self.channels})"


class StimDesign:
    def __init__(self, *args):
        # Pairs of (duration_us, current_uA)
        assert len(args) % 2 == 0, "StimDesign needs (duration, current) pairs"
        self.phases = [
            (args[i], args[i + 1]) for i in range(0, len(args), 2)
        ]
        self.total_charge = sum(dur * cur for dur, cur in self.phases)

    @property
    def amplitude(self) -> float:
        return max(abs(cur) for _, cur in self.phases)

    def __repr__(self):
        return f"StimDesign({self.phases})"


class BurstDesign:
    def __init__(self, count: int, frequency_hz: float):
        self.count = count
        self.frequency_hz = frequency_hz


# ── Simulated Neurons (the reservoir) ──────────────────────────────────────

class Neurons:
    """
    Simulates the biological neural reservoir on the CL1 MEA.

    Internally, each electrode has a random "connectivity weight" vector that
    determines how it responds to stimulation on other electrodes.  This is
    the mock version of the actual recurrent biological network formed by
    cultured neurons sitting on the array.
    """

    def __init__(self, seed: Optional[int] = None):
        self._rng = np.random.RandomState(seed)
        self._timestamp = 0

        # ── Reservoir weights (fixed random projection) ──────────────
        # W_in: how each active channel responds to stimulation input.
        # Shape: (n_active_channels, n_active_channels)
        # This models the synaptic connectivity between neurons.
        n = len(ACTIVE_CHANNELS)
        self._W_in = self._rng.randn(n, n) * 0.6
        # Add some sparsity -- biological networks aren't fully connected.
        mask = self._rng.rand(n, n) < 0.35  # ~35 % connectivity
        self._W_in *= mask

        # Internal reservoir state (membrane potentials, roughly)
        self._state = np.zeros(n)

        # Channel index map: electrode number → index into active array
        self._ch_to_idx = {ch: i for i, ch in enumerate(ACTIVE_CHANNELS)}

        # Last stim vector (used to compute response in next loop tick)
        self._pending_stim = np.zeros(n)

    def timestamp(self) -> int:
        return self._timestamp

    def get_frames_per_second(self) -> int:
        return FRAMES_PER_SECOND

    # ── Stimulation ──────────────────────────────────────────────────

    def stim(self, channels, stim_design, burst_design=None):
        # Accept int shorthand like the real API
        if isinstance(channels, int):
            channels = ChannelSet(channels)
            if isinstance(stim_design, (int, float)):
                stim_design = StimDesign(180, -stim_design, 180, stim_design)

        amplitude = stim_design.amplitude
        repeats = burst_design.count if burst_design else 1

        for ch in channels:
            if ch in self._ch_to_idx:
                idx = self._ch_to_idx[ch]
                self._pending_stim[idx] += amplitude * repeats

    # ── The loop (closed-loop control) ───────────────────────────────

    def loop(self, ticks_per_second: int, *,
             stop_after_seconds: float = None,
             stop_after_ticks: int = None,
             ignore_jitter: bool = True,
             jitter_tolerance_frames: int = 0):

        frames_per_tick = FRAMES_PER_SECOND // ticks_per_second
        if stop_after_ticks is not None:
            total = stop_after_ticks
        elif stop_after_seconds is not None:
            total = int(stop_after_seconds * ticks_per_second)
        else:
            total = 10 * ticks_per_second  # default 10 s safety cap

        for i in range(total):
            self._timestamp += frames_per_tick

            # ── Reservoir dynamics step ──────────────────────────────
            # Leaky integrator model (standard echo state network):
            # x(t+1) = (1-a)*x(t) + a*tanh(W @ x(t) + stim + noise)
            # The leak rate `a` models the membrane time constant.
            leak = 0.3
            drive = self._W_in @ self._state + self._pending_stim
            noise = self._rng.randn(len(ACTIVE_CHANNELS)) * 0.05
            self._state = (1 - leak) * self._state + leak * np.tanh(drive + noise)
            self._pending_stim[:] = 0.0

            # ── Generate spikes from reservoir state ─────────────────
            # Probability of spiking ∝ sigmoid of state magnitude.
            spike_probs = 1.0 / (1.0 + np.exp(-8.0 * (np.abs(self._state) - 0.25)))
            firing = self._rng.rand(len(ACTIVE_CHANNELS)) < spike_probs

            spikes = []
            for idx in np.where(firing)[0]:
                spikes.append(Spike(
                    channel=ACTIVE_CHANNELS[idx],
                    timestamp=self._timestamp + self._rng.randint(0, frames_per_tick),
                ))

            yield Tick(
                iteration=i,
                timestamp=self._timestamp,
                analysis=TickAnalysis(spikes=spikes),
            )

    # ── Raw read (for completeness) ──────────────────────────────────

    def read(self, num_frames: int, from_timestamp: int) -> np.ndarray:
        # Returns synthetic raw voltage traces shaped (num_frames, 64)
        data = self._rng.randn(num_frames, TOTAL_CHANNELS).astype(np.float32) * 10.0
        # Inject larger signals on channels with high reservoir state
        for idx, ch in enumerate(ACTIVE_CHANNELS):
            data[:, ch] += self._state[idx] * 50.0
        return (data / 0.195).astype(np.int16)  # convert µV → raw int16


# ── Context manager (mirrors cl.open()) ─────────────────────────────────────

@contextmanager
def open(seed: Optional[int] = None):
    neurons = Neurons(seed=seed)
    try:
        yield neurons
    finally:
        pass  # real API releases hardware here
