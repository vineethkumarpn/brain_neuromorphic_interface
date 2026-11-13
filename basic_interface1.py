"""Small, dependency-free utilities for exploring neural signals and spiking models.

Signals are represented as sequences of floats. EEG data is channel-major:
``channels[channel][sample]``. These examples are for learning and simulation,
not for clinical use.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _as_finite_signal(signal: Sequence[float], name: str = "signal") -> list[float]:
    """Return a non-empty signal as floats, rejecting invalid sample values."""
    if not signal:
        raise ValueError(f"{name} must contain at least one sample")

    values = [float(sample) for sample in signal]
    if not all(math.isfinite(sample) for sample in values):
        raise ValueError(f"{name} samples must all be finite")
    return values


def validate_eeg(
    channels: Sequence[Sequence[float]], sampling_rate_hz: float
) -> None:
    """Validate channel-major EEG samples and their sampling rate."""
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero")
    if not channels:
        raise ValueError("channels must contain at least one channel")

    sample_count: int | None = None
    for channel_index, channel in enumerate(channels):
        values = _as_finite_signal(channel, f"channel {channel_index}")
        if sample_count is None:
            sample_count = len(values)
        elif len(values) != sample_count:
            raise ValueError("all EEG channels must have the same number of samples")


def remove_dc(signal: Sequence[float]) -> list[float]:
    """Subtract a signal's mean to remove its constant (DC) offset."""
    values = _as_finite_signal(signal)
    mean = sum(values) / len(values)
    return [sample - mean for sample in values]


def zscore(signal: Sequence[float]) -> list[float]:
    """Standardize samples to zero mean and unit population standard deviation."""
    values = _as_finite_signal(signal)
    mean = sum(values) / len(values)
    variance = sum((sample - mean) ** 2 for sample in values) / len(values)
    standard_deviation = math.sqrt(variance)
    if standard_deviation == 0:
        raise ValueError("cannot z-score a constant signal")
    return [(sample - mean) / standard_deviation for sample in values]


def sliding_windows(
    signal: Sequence[float], window_size: int, step_size: int
) -> list[list[float]]:
    """Split a signal into fixed-size windows, dropping any incomplete tail."""
    values = _as_finite_signal(signal)
    if (
        isinstance(window_size, bool)
        or isinstance(step_size, bool)
        or not isinstance(window_size, int)
        or not isinstance(step_size, int)
    ):
        raise ValueError("window_size and step_size must be integers")
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window_size and step_size must be positive")

    return [
        values[start : start + window_size]
        for start in range(0, len(values) - window_size + 1, step_size)
    ]


def threshold_crossings(
    signal: Sequence[float], threshold: float, refractory_samples: int = 0
) -> list[int]:
    """Return indices where a signal first rises above threshold.

    A crossing at index zero is included when the first sample is already above
    threshold. After a crossing, further crossings are ignored for the given
    number of samples.
    """
    values = _as_finite_signal(signal)
    if not math.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if isinstance(refractory_samples, bool) or not isinstance(refractory_samples, int):
        raise ValueError("refractory_samples must be a non-negative integer")
    if refractory_samples < 0:
        raise ValueError("refractory_samples must be a non-negative integer")

    crossings: list[int] = []
    last_crossing = -refractory_samples - 1
    previous = -math.inf
    for index, sample in enumerate(values):
        if sample > threshold and previous <= threshold:
            if index - last_crossing > refractory_samples:
                crossings.append(index)
                last_crossing = index
        previous = sample
    return crossings


def encode_spikes(
    signal: Sequence[float], threshold: float, refractory_samples: int = 0
) -> list[int]:
    """Encode threshold crossings as a binary spike train (one value per sample)."""
    values = _as_finite_signal(signal)
    spike_train = [0] * len(values)
    for index in threshold_crossings(values, threshold, refractory_samples):
        spike_train[index] = 1
    return spike_train


def inter_spike_intervals(spike_times_ms: Sequence[float]) -> list[float]:
    """Return time differences between ordered, non-negative spike timestamps."""
    times = [float(time_ms) for time_ms in spike_times_ms]
    if any(not math.isfinite(time_ms) or time_ms < 0 for time_ms in times):
        raise ValueError("spike times must be finite and non-negative")
    if any(later <= earlier for earlier, later in zip(times, times[1:])):
        raise ValueError("spike times must be strictly increasing")
    return [later - earlier for earlier, later in zip(times, times[1:])]


def firing_rate_hz(spike_times_ms: Sequence[float], duration_ms: float) -> float:
    """Compute mean firing rate in hertz over a positive observation duration."""
    if not math.isfinite(duration_ms) or duration_ms <= 0:
        raise ValueError("duration_ms must be finite and greater than zero")
    inter_spike_intervals(spike_times_ms)
    times = [float(time_ms) for time_ms in spike_times_ms]
    if times and times[-1] > duration_ms:
        raise ValueError("spike times must fall within the observation duration")
    return len(times) * 1000.0 / duration_ms


def simulate_lif(
    input_current_nA: Sequence[float],
    sampling_rate_hz: float,
    *,
    time_constant_ms: float = 20.0,
    resistance_mohm: float = 100.0,
    resting_potential_mV: float = -65.0,
    threshold_mV: float = -50.0,
    reset_potential_mV: float = -65.0,
    refractory_period_ms: float = 2.0,
) -> tuple[list[float], list[int]]:
    """Simulate a leaky integrate-and-fire neuron with forward Euler steps.

    Current is in nA, resistance in MOhm, voltage in mV, and time in ms.
    Returns the voltage trace and spike sample indices.
    """
    currents = _as_finite_signal(input_current_nA, "input_current_nA")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero")
    parameters = (
        time_constant_ms,
        resistance_mohm,
        resting_potential_mV,
        threshold_mV,
        reset_potential_mV,
        refractory_period_ms,
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("model parameters must all be finite")
    if time_constant_ms <= 0 or resistance_mohm <= 0 or refractory_period_ms < 0:
        raise ValueError(
            "time_constant_ms and resistance_mohm must be positive; "
            "refractory_period_ms must be non-negative"
        )
    if reset_potential_mV >= threshold_mV:
        raise ValueError("reset_potential_mV must be below threshold_mV")

    step_ms = 1000.0 / sampling_rate_hz
    refractory_steps = math.ceil(refractory_period_ms / step_ms)
    remaining_refractory_steps = 0
    voltage = resting_potential_mV
    voltage_trace: list[float] = []
    spike_indices: list[int] = []

    for index, current_nA in enumerate(currents):
        if remaining_refractory_steps:
            voltage = reset_potential_mV
            remaining_refractory_steps -= 1
        else:
            voltage += (
                step_ms
                / time_constant_ms
                * (
                    resting_potential_mV
                    - voltage
                    + resistance_mohm * current_nA
                )
            )
            if voltage >= threshold_mV:
                spike_indices.append(index)
                voltage = reset_potential_mV
                remaining_refractory_steps = refractory_steps
        voltage_trace.append(voltage)

    return voltage_trace, spike_indices


def decode_population_direction(
    spike_counts: Sequence[float], preferred_directions_rad: Sequence[float]
) -> float:
    """Decode a circular population response as an angle in radians.

    Each count weights a neuron's preferred direction. Raises ``ValueError`` if
    the population has no activity or its weighted direction is undefined.
    """
    counts = _as_finite_signal(spike_counts, "spike_counts")
    directions = _as_finite_signal(
        preferred_directions_rad, "preferred_directions_rad"
    )
    if len(counts) != len(directions):
        raise ValueError("spike counts and preferred directions must have equal lengths")
    if any(count < 0 for count in counts):
        raise ValueError("spike counts must be non-negative")

    x = sum(count * math.cos(direction) for count, direction in zip(counts, directions))
    y = sum(count * math.sin(direction) for count, direction in zip(counts, directions))
    if math.hypot(x, y) <= 1e-12:
        raise ValueError("population activity does not define a unique direction")
    return math.atan2(y, x)


def main() -> None:
    """Run a small synthetic example through each starter utility."""
    sampling_rate_hz = 100.0
    eeg_channels = [
        [-1.0, 0.0, 1.0, 0.0, 2.0, 0.0, -1.0, 0.0],
        [0.2, -0.3, 0.5, 0.1, -0.4, 0.6, -0.2, 0.0],
    ]
    validate_eeg(eeg_channels, sampling_rate_hz)

    channel = eeg_channels[0]
    centered_signal = remove_dc(channel)
    standardized_signal = zscore(channel)
    windows = sliding_windows(channel, window_size=4, step_size=2)
    crossing_indices = threshold_crossings(centered_signal, threshold=0.5)
    spike_train = encode_spikes(centered_signal, threshold=0.5)

    sample_interval_ms = 1000.0 / sampling_rate_hz
    spike_times_ms = [index * sample_interval_ms for index in crossing_indices]
    intervals_ms = inter_spike_intervals(spike_times_ms)
    rate = firing_rate_hz(spike_times_ms, duration_ms=80.0)

    voltage_trace, lif_spikes = simulate_lif([0.5] * 20, sampling_rate_hz=1000.0)
    decoded_direction_rad = decode_population_direction(
        spike_counts=[5.0, 2.0, 1.0],
        preferred_directions_rad=[0.0, math.pi / 2, math.pi],
    )

    print(f"Validated {len(eeg_channels)} EEG channels at {sampling_rate_hz:g} Hz")
    print(f"DC-removed samples: {centered_signal}")
    print(f"Z-scored samples: {standardized_signal}")
    print(f"Overlapping windows: {windows}")
    print(f"Threshold crossing indices: {crossing_indices}")
    print(f"Encoded spike train: {spike_train}")
    print(f"Inter-spike intervals (ms): {intervals_ms}")
    print(f"Threshold-event firing rate: {rate:.1f} Hz")
    print(f"LIF voltage samples: {len(voltage_trace)}, spike indices: {lif_spikes}")
    print(f"Decoded population direction: {decoded_direction_rad:.3f} rad")


if __name__ == "__main__":
    main()
