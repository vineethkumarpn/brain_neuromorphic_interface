# brain_neuromorphic_interface

An educational project for exploring neural signals, spiking-neuron models, and
brain-computer interface concepts. The examples are for research and learning,
not for clinical use.

## Getting started

The workspace virtual environment is `.venv`. Activate it in a terminal with:

```sh
source .venv/bin/activate
```

The starter module uses only the Python standard library. Its functions are in
[`basic_interface1.py`](basic_interface1.py). Run its synthetic demonstration
with:

```sh
python basic_interface1.py
```

## Starter functions

- `validate_eeg`: check channel-major EEG data and its sampling rate.
- `remove_dc`, `zscore`: basic signal centering and standardization.
- `sliding_windows`: split a signal into fixed-size analysis windows.
- `threshold_crossings`, `encode_spikes`: convert threshold crossings to event
  indices or a binary spike train.
- `inter_spike_intervals`, `firing_rate_hz`: summarize spike timing.
- `simulate_lif`: simulate a leaky integrate-and-fire neuron.
- `decode_population_direction`: decode a circular direction from population
  spike counts and preferred directions in radians.

For example:

```python
from basic_interface1 import encode_spikes, firing_rate_hz

spike_train = encode_spikes([-1.0, 0.2, 1.2, 0.4, 1.5], threshold=1.0)
spike_times_ms = [10.0, 40.0]
rate = firing_rate_hz(spike_times_ms, duration_ms=100.0)
```

EEG arrays are represented as `channels[channel][sample]`; sampling rates use
hertz, timestamps use milliseconds, and the LIF model uses nA, MOhm, and mV.

This is a solo research project by P N Vineeth Kumar.

## Optional AI integration

[`ai_integ.py`](ai_integ.py) provides model-assisted utilities without tying
the project to a particular AI vendor. Supply a callable that accepts a text
prompt and returns the model response as a JSON string. The module itself does
not make network requests or install additional packages.

```python
from ai_integ import (
    generate_demo_inputs,
    optimize_lif_parameters,
    summarize_results_with_ai,
)

# Replace this function with an adapter for your chosen local or hosted model.
def ask_model(prompt: str) -> str:
    return your_model_client.generate_text(prompt)

generated = generate_demo_inputs(ask_model, sample_count=64)
search = optimize_lif_parameters(
    ask_model,
    generated["input_current_nA"],
    generated["sampling_rate_hz"],
    target_rate_hz=10.0,
)
explanation = summarize_results_with_ai(
    ask_model,
    {
        "target_rate_hz": search["target_rate_hz"],
        "best_rate_hz": search["best_candidate"]["firing_rate_hz"],
    },
)
```

`run_generated_demo(ask_model)` also generates synthetic inputs and runs them
through the starter signal-processing and neuron functions. The LIF search
reports the best among a bounded set of model-proposed candidates; it does not
guarantee a global optimum. Validate model output and use synthetic or
de-identified data. These educational examples are not for clinical use or
controlling a device.
