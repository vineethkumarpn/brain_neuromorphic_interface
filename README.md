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
