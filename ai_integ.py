"""Provider-neutral AI helpers for the neuromorphic starter functions.

Pass a callable that accepts a prompt and returns the model's text response.
The callable can wrap a local model or a hosted provider; this module does not
send data anywhere on its own and installs no model-specific dependencies.
Use synthetic or de-identified data, and treat model suggestions as exploratory
simulation results rather than clinical recommendations.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TypedDict

from basic_interface1 import (
    encode_spikes,
    firing_rate_hz,
    inter_spike_intervals,
    remove_dc,
    simulate_lif,
    sliding_windows,
    threshold_crossings,
    validate_eeg,
    zscore,
)

ModelTextGenerator = Callable[[str], str]


class DemoInputs(TypedDict):
    """Validated synthetic values that can be passed to the starter functions."""

    sampling_rate_hz: float
    channels: list[list[float]]
    input_current_nA: list[float]
    threshold: float
    window_size: int
    step_size: int


class LIFParameters(TypedDict):
    """Parameters varied by the model-assisted LIF search."""

    time_constant_ms: float
    resistance_mohm: float
    resting_potential_mV: float
    threshold_mV: float
    reset_potential_mV: float
    refractory_period_ms: float


class GeneratedDemoResult(TypedDict):
    inputs: DemoInputs
    dc_removed_sample: list[float]
    dc_removed_signal: list[float]
    zscored_sample: list[float]
    zscored_signal: list[float]
    window_count: int
    threshold_crossing_indices: list[int]
    encoded_spike_count: int
    lif_spike_indices: list[int]
    lif_firing_rate_hz: float
    lif_inter_spike_intervals_ms: list[float]
    lif_voltage_sample_mV: list[float]
    lif_voltage_trace: list[float]


def _request_json_object(
    model: ModelTextGenerator, prompt: str
) -> dict[str, object]:
    """Call a model and require a JSON object response."""
    response = model(prompt)
    if not isinstance(response, str):
        raise TypeError("model callable must return a JSON string")
    try:
        result = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("model response was not valid JSON") from error
    if not isinstance(result, dict):
        raise ValueError("model response must be a JSON object")
    return result


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def generate_demo_inputs(
    model: ModelTextGenerator, *, sample_count: int = 64
) -> DemoInputs:
    """Ask a model for bounded synthetic EEG/current values for exploration.

    The model must return JSON with ``sampling_rate_hz``, exactly two
    ``channels`` each containing ``sample_count`` numbers, an
    ``input_current_nA`` list of that length, ``threshold``, ``window_size``,
    and ``step_size``. Numeric ranges and array shapes are validated before
    the values are returned.
    """
    sample_count = _positive_integer(sample_count, "sample_count")
    if sample_count > 10_000:
        raise ValueError("sample_count must be no greater than 10000")

    payload = _request_json_object(
        model,
        (
            "Create a small synthetic dataset for an educational EEG and "
            "spiking-neuron simulation. Return only a JSON object with these "
            "keys: sampling_rate_hz, channels, input_current_nA, threshold, "
            "window_size, step_size. channels must be exactly two arrays, "
            f"each with {sample_count} finite numeric samples. "
            f"input_current_nA must have {sample_count} finite numeric "
            "samples. Keep channel values between -100 and 100 arbitrary "
            "units, currents between -5 and 5 nA, sampling_rate_hz between "
            "50 and 100000 Hz, and choose a positive threshold and window "
            "sizes that fit the data. Use no personal or clinical data."
        ),
    )

    sampling_rate_hz = _finite_number(
        payload.get("sampling_rate_hz"), "sampling_rate_hz"
    )
    if not 50 <= sampling_rate_hz <= 100_000:
        raise ValueError("sampling_rate_hz must be between 50 and 100000")

    raw_channels = payload.get("channels")
    if not isinstance(raw_channels, list) or len(raw_channels) != 2:
        raise ValueError("channels must contain exactly two arrays")
    channels: list[list[float]] = []
    for channel_index, raw_channel in enumerate(raw_channels):
        if not isinstance(raw_channel, list) or len(raw_channel) != sample_count:
            raise ValueError(
                f"channel {channel_index} must contain exactly {sample_count} samples"
            )
        channel = [
            _finite_number(value, f"channel {channel_index} sample")
            for value in raw_channel
        ]
        if any(abs(value) > 100 for value in channel):
            raise ValueError("channel samples must be between -100 and 100")
        channels.append(channel)
    if len(set(channels[0])) < 2:
        raise ValueError("the first channel must vary so it can be z-scored")

    raw_current = payload.get("input_current_nA")
    if not isinstance(raw_current, list) or len(raw_current) != sample_count:
        raise ValueError(f"input_current_nA must contain exactly {sample_count} samples")
    input_current_nA = [
        _finite_number(value, "input_current_nA sample") for value in raw_current
    ]
    if any(abs(value) > 5 for value in input_current_nA):
        raise ValueError("input currents must be between -5 and 5 nA")

    threshold = _finite_number(payload.get("threshold"), "threshold")
    window_size = _positive_integer(payload.get("window_size"), "window_size")
    step_size = _positive_integer(payload.get("step_size"), "step_size")
    if window_size > sample_count or step_size > sample_count:
        raise ValueError("window_size and step_size must not exceed sample_count")
    validate_eeg(channels, sampling_rate_hz)

    return {
        "sampling_rate_hz": sampling_rate_hz,
        "channels": channels,
        "input_current_nA": input_current_nA,
        "threshold": threshold,
        "window_size": window_size,
        "step_size": step_size,
    }


def run_generated_demo(
    model: ModelTextGenerator, *, sample_count: int = 64
) -> GeneratedDemoResult:
    """Generate synthetic inputs and return signal traces and summary results."""
    inputs = generate_demo_inputs(model, sample_count=sample_count)
    channel = inputs["channels"][0]
    centered = remove_dc(channel)
    standardized = zscore(channel)
    windows = sliding_windows(channel, inputs["window_size"], inputs["step_size"])
    crossing_indices = threshold_crossings(centered, inputs["threshold"])
    spike_train = encode_spikes(centered, inputs["threshold"])

    voltage_trace, lif_spike_indices = simulate_lif(
        inputs["input_current_nA"], inputs["sampling_rate_hz"]
    )
    duration_ms = sample_count * 1000.0 / inputs["sampling_rate_hz"]
    lif_spike_times_ms = [
        index * 1000.0 / inputs["sampling_rate_hz"] for index in lif_spike_indices
    ]

    return {
        "inputs": inputs,
        "dc_removed_sample": centered[:10],
        "dc_removed_signal": centered,
        "zscored_sample": standardized[:10],
        "zscored_signal": standardized,
        "window_count": len(windows),
        "threshold_crossing_indices": crossing_indices,
        "encoded_spike_count": sum(spike_train),
        "lif_spike_indices": lif_spike_indices,
        "lif_firing_rate_hz": firing_rate_hz(lif_spike_times_ms, duration_ms),
        "lif_inter_spike_intervals_ms": inter_spike_intervals(lif_spike_times_ms),
        "lif_voltage_sample_mV": voltage_trace[:10],
        "lif_voltage_trace": voltage_trace,
    }


def save_generated_demo_plot(
    model: ModelTextGenerator,
    output_path: str | Path = "generated_demo.png",
    *,
    sample_count: int = 64,
) -> Path:
    """Save generated demo inputs and computed output traces to a PNG image."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    results = run_generated_demo(model, sample_count=sample_count)
    inputs = results["inputs"]
    sampling_rate_hz = inputs["sampling_rate_hz"]
    sample_times_ms = [
        index * 1000.0 / sampling_rate_hz
        for index in range(len(inputs["input_current_nA"]))
    ]

    figure = Figure(figsize=(13, 10), constrained_layout=True)
    FigureCanvasAgg(figure)
    axes = figure.subplots(3, 2)
    figure.suptitle("Synthetic neuromorphic demo: inputs and outputs")

    channel_axis = axes[0, 0]
    channel_axis.plot(sample_times_ms, inputs["channels"][0], label="Channel 1")
    channel_axis.plot(sample_times_ms, inputs["channels"][1], label="Channel 2")
    channel_axis.set_title("Input EEG channels")
    channel_axis.set_ylabel("Amplitude (arbitrary units)")
    channel_axis.legend()

    current_axis = axes[0, 1]
    current_axis.plot(sample_times_ms, inputs["input_current_nA"], color="tab:orange")
    current_axis.set_title("Input current")
    current_axis.set_ylabel("Current (nA)")

    centered_axis = axes[1, 0]
    centered_signal = results["dc_removed_signal"]
    centered_axis.plot(sample_times_ms, centered_signal, label="DC removed")
    centered_axis.axhline(
        inputs["threshold"], color="tab:red", linestyle="--", label="Threshold"
    )
    crossing_indices = results["threshold_crossing_indices"]
    centered_axis.scatter(
        [sample_times_ms[index] for index in crossing_indices],
        [centered_signal[index] for index in crossing_indices],
        color="tab:red",
        marker="x",
        label="Threshold crossings",
    )
    centered_axis.set_title("Processed channel")
    centered_axis.set_ylabel("Amplitude")
    centered_axis.legend()

    standardized_axis = axes[1, 1]
    standardized_axis.plot(sample_times_ms, results["zscored_signal"])
    standardized_axis.set_title("Z-scored channel")
    standardized_axis.set_ylabel("Standard deviations")

    voltage_axis = axes[2, 0]
    voltage_trace = results["lif_voltage_trace"]
    voltage_axis.plot(sample_times_ms, voltage_trace, label="LIF voltage")
    spike_indices = results["lif_spike_indices"]
    voltage_axis.scatter(
        [sample_times_ms[index] for index in spike_indices],
        [voltage_trace[index] for index in spike_indices],
        color="tab:red",
        marker="x",
        label="LIF spikes",
    )
    voltage_axis.set_title("Leaky integrate-and-fire output")
    voltage_axis.set_xlabel("Time (ms)")
    voltage_axis.set_ylabel("Voltage (mV)")
    voltage_axis.legend()

    summary_axis = axes[2, 1]
    summary_axis.axis("off")
    intervals = results["lif_inter_spike_intervals_ms"]
    interval_lines = [f"  {interval:.2f}" for interval in intervals[:6]]
    if len(intervals) > 6:
        interval_lines.append(f"  ... ({len(intervals)} intervals total)")
    interval_text = "\n".join(interval_lines) or "  None"
    summary_axis.text(
        0.02,
        0.98,
        "\n".join(
            (
                "Demo summary",
                f"Sampling rate: {sampling_rate_hz:g} Hz",
                f"Window count: {results['window_count']}",
                f"Threshold crossings: {len(crossing_indices)}",
                f"Encoded spikes: {results['encoded_spike_count']}",
                f"LIF spikes: {len(spike_indices)}",
                f"LIF firing rate: {results['lif_firing_rate_hz']:.2f} Hz",
                "LIF inter-spike intervals (ms):",
                interval_text,
            )
        ),
        va="top",
    )

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("output_path must have a .png extension")
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="png", dpi=150)
    return path


def optimize_lif_parameters(
    model: ModelTextGenerator,
    input_current_nA: Sequence[float],
    sampling_rate_hz: float,
    target_rate_hz: float,
    *,
    max_candidates: int = 8,
) -> dict[str, object]:
    """Evaluate AI-proposed LIF settings and return the closest simulated rate.

    This is a bounded candidate search, not a proof of globally optimal
    parameters. Every suggestion is checked by ``simulate_lif`` and scored
    locally by absolute error from ``target_rate_hz``.
    """
    current = [
        _finite_number(value, "input_current_nA sample")
        for value in input_current_nA
    ]
    if not current:
        raise ValueError("input_current_nA must contain at least one sample")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be finite and greater than zero")
    if not math.isfinite(target_rate_hz) or target_rate_hz < 0:
        raise ValueError("target_rate_hz must be finite and non-negative")
    max_candidates = _positive_integer(max_candidates, "max_candidates")
    if max_candidates > 20:
        raise ValueError("max_candidates must be no greater than 20")

    adjustable_parameters = (
        "time_constant_ms",
        "resistance_mohm",
        "threshold_mV",
        "refractory_period_ms",
    )
    prompt = (
        "Suggest candidate parameters for the educational leaky integrate-and-fire "
        "simulation described below. Return only JSON of the form "
        '{"candidates":[{"time_constant_ms":20,"resistance_mohm":100,'
        '"threshold_mV":-50,"refractory_period_ms":2}]}. Include between 1 and '
        f"{max_candidates} candidates, only these adjustable keys: "
        f"{', '.join(adjustable_parameters)}. Values must be finite; "
        "time_constant_ms and resistance_mohm must be positive, threshold_mV "
        "must be above the fixed reset potential -65 mV, refractory_period_ms "
        "must be non-negative. Propose simulation settings only, not medical "
        "or device-control recommendations.\n"
        f"sampling_rate_hz: {sampling_rate_hz}\n"
        f"target_rate_hz: {target_rate_hz}\n"
        f"input_current_nA: {json.dumps(current)}"
    )
    response = _request_json_object(model, prompt)
    raw_candidates = response.get("candidates")
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= max_candidates:
        raise ValueError(f"model must return between 1 and {max_candidates} candidates")

    candidate_results: list[dict[str, object]] = []
    for candidate_index, raw_candidate in enumerate(raw_candidates):
        if not isinstance(raw_candidate, dict):
            raise ValueError(f"candidate {candidate_index} must be a JSON object")
        if set(raw_candidate) != set(adjustable_parameters):
            raise ValueError(
                f"candidate {candidate_index} must provide exactly "
                f"{', '.join(adjustable_parameters)}"
            )
        candidate_values = {
            key: _finite_number(raw_candidate[key], f"candidate {candidate_index} {key}")
            for key in adjustable_parameters
        }
        parameters: LIFParameters = {
            "time_constant_ms": candidate_values["time_constant_ms"],
            "resistance_mohm": candidate_values["resistance_mohm"],
            "resting_potential_mV": -65.0,
            "threshold_mV": candidate_values["threshold_mV"],
            "reset_potential_mV": -65.0,
            "refractory_period_ms": candidate_values["refractory_period_ms"],
        }
        voltage_trace, spike_indices = simulate_lif(
            current, sampling_rate_hz, **parameters
        )
        duration_ms = len(current) * 1000.0 / sampling_rate_hz
        spike_times_ms = [
            index * 1000.0 / sampling_rate_hz for index in spike_indices
        ]
        rate = firing_rate_hz(spike_times_ms, duration_ms)
        candidate_results.append(
            {
                "parameters": parameters,
                "firing_rate_hz": rate,
                "absolute_rate_error_hz": abs(rate - target_rate_hz),
                "spike_count": len(spike_indices),
                "voltage_sample_mV": voltage_trace[:10],
            }
        )

    best = min(
        candidate_results,
        key=lambda result: float(result["absolute_rate_error_hz"]),
    )
    return {
        "target_rate_hz": target_rate_hz,
        "best_candidate": best,
        "evaluated_candidates": candidate_results,
        "optimization_note": (
            "Best of the model-proposed candidates under this simulation and "
            "objective; not guaranteed globally optimal."
        ),
    }


def summarize_results_with_ai(
    model: ModelTextGenerator, metrics: Mapping[str, int | float]
) -> dict[str, object]:
    """Ask a model to explain numeric simulation metrics and suggest next checks.

    Only the explicitly supplied metric mapping is sent to the model. Raw
    signals and identifying data should not be included.
    """
    if not metrics:
        raise ValueError("metrics must contain at least one numeric value")
    sanitized_metrics: dict[str, int | float] = {}
    for name, value in metrics.items():
        if not isinstance(name, str) or not name:
            raise ValueError("metric names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"metric {name!r} must be numeric")
        if not math.isfinite(value):
            raise ValueError(f"metric {name!r} must be finite")
        sanitized_metrics[name] = value

    payload = _request_json_object(
        model,
        (
            "Explain these numeric outputs from an educational neural-signal "
            "simulation. Do not diagnose a person or make clinical/device-control "
            "recommendations. State uncertainty, and suggest analysis checks "
            "rather than claiming causation. Return only JSON with a string "
            '"summary", an array of string "observations", and an array of '
            '"next_steps".\n'
            f"Metrics: {json.dumps(sanitized_metrics, allow_nan=False)}"
        ),
    )
    summary = payload.get("summary")
    observations = payload.get("observations")
    next_steps = payload.get("next_steps")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("model summary must be a non-empty string")
    if not isinstance(observations, list) or not all(
        isinstance(item, str) for item in observations
    ):
        raise ValueError("model observations must be an array of strings")
    if not isinstance(next_steps, list) or not all(
        isinstance(item, str) for item in next_steps
    ):
        raise ValueError("model next_steps must be an array of strings")
    return {
        "summary": summary.strip(),
        "observations": observations,
        "next_steps": next_steps,
    }