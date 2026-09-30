"""Observation model: discretisation, noise, and realistic missingness."""

from __future__ import annotations

from typing import Callable

import numpy as np


class ObservationModel:
    """Apply sampling cadence, noise, and dropouts to latent continuous signals."""

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng or np.random.default_rng()

    def observe(
        self,
        t: np.ndarray,
        signal: np.ndarray,
        sampling_interval: float,
        noise_sd: float,
        dropout_rate: float = 0.0,
        value_range: tuple[float, float] | None = None,
        hold: bool = False,
        gap_rate: float = 0.0,
        gap_duration_s: float = 30.0,
        timing_jitter_frac: float = 0.0,
    ) -> np.ndarray:
        """Return an observed array with the same shape as `signal`.

        Values are sampled at `sampling_interval` and corrupted with Gaussian noise.
        Between samples the value is NaN (matching real VitalDB cadence) unless `hold=True`.
        Optional `gap_rate` creates realistic multi-second dropouts (e.g. probe disconnection).

        `timing_jitter_frac` adds uniform jitter ±frac to each inter-sample interval,
        reproducing the non-zero dt_std seen in real monitor recordings.  When 0 the
        behaviour is identical to the original implementation.
        """
        observed = np.full_like(signal, np.nan)
        last_sample_idx = 0
        gap_end = -1.0
        # Time-based clock: track when the *next* sample is due rather than the
        # index of the last sample.  This lets timing_jitter_frac perturb each
        # inter-sample interval independently.
        next_due: float = float(t[0])

        def _advance_clock(t_now: float) -> float:
            interval = sampling_interval
            if timing_jitter_frac > 0.0:
                interval *= 1.0 + self.rng.uniform(-timing_jitter_frac, timing_jitter_frac)
            return t_now + interval

        for i in range(len(t)):
            if t[i] < gap_end:
                continue
            if t[i] >= next_due - 1e-9:
                if self.rng.random() < gap_rate:
                    gap_end = t[i] + self.rng.exponential(gap_duration_s)
                    # Advance the sampling clock even though no value is recorded.
                    next_due = _advance_clock(t[i])
                    continue
                if self.rng.random() >= dropout_rate:
                    observed[i] = signal[i] + self.rng.normal(0, noise_sd)
                # Update the sampling clock regardless of dropout; the next sample is
                # still due at the nominal cadence. This matches real monitor on/off behavior.
                next_due = _advance_clock(t[i])
                last_sample_idx = i
            elif hold:
                observed[i] = observed[last_sample_idx]

        if value_range is not None:
            low, high = value_range
            observed = np.clip(observed, low, high)
        return observed

    def observe_discrete_event(
        self,
        t: np.ndarray,
        event_times: list[float],
        event_values: list[float],
        hold: bool = True,
        dropout_rate: float = 0.0,
    ) -> np.ndarray:
        """Observe a stepwise signal (e.g. infusion rate) from event times."""
        out = np.full_like(t, np.nan) if dropout_rate > 0 else np.zeros_like(t, dtype=float)
        for i, ti in enumerate(t):
            # Find last event at or before ti
            value = 0.0
            for et, ev in zip(event_times, event_values):
                if et <= ti:
                    value = ev
                else:
                    break
            if dropout_rate > 0:
                if self.rng.random() >= dropout_rate:
                    out[i] = value
                elif hold and i > 0:
                    out[i] = out[i - 1]
            else:
                out[i] = value
        return out
