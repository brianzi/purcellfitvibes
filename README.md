# purcellfitvibes

Experimental Python code for identifying and fitting multiplexed Purcell-filter/readout structures from chirp measurements, motivated by the architecture and input-output model in [Jeffrey et al., arXiv:1801.07904](https://arxiv.org/abs/1801.07904).

The intended workflow is:

1. send one fast broadband chirp through the common feedline;
2. estimate the complex transfer function on well-excited Fourier bins;
3. detect broad spectral features and fit each with a **generic two-pole rational model**;
4. forget the feature labels, treat the poles as an unordered set, and pair them with a minimum-cost matching heuristic;
5. use each pole pair to choose the center/bandwidth of a targeted chirp;
6. fit the targeted trace with the physical Purcell/readout model (four nonlinear device parameters plus delay), profiling out complex gain/background nuisance terms.

The broadband stage is intentionally not a physical fit. Its job is topology/feature discovery and robust initialization. The narrow stage does the parameter estimation.

## Install

```bash
python -m pip install -e '.[test]'
pytest
```

## Demo

```bash
python examples/end_to_end.py
```

The demo generates five synthetic structures, one 300 ns broadband chirp spanning about 1.1 GHz, extracts ten generic poles, pairs them into five structures, simulates targeted scans, performs physical fits, prints the recovered parameters, and writes `broad_scan.png`.

## Package layout

- `model.py` — single- and multi-structure transmission models.
- `signals.py` — finite-duration smoothly windowed chirps.
- `sim.py` — synthetic feedline traces with delay, gain slope, noise and optional echo.
- `poles.py` — broadband transfer estimate, feature detection and generic two-pole rational fits.
- `pairing.py` — minimum-cost perfect matching of an unordered pole set.
- `fitting.py` — pole-pair initialization and targeted physical trace fitting.

## Model limitations

The multiplexed model treats the structures as independent side-coupled scatterers on a shared through line. It does **not** yet model coherent multiple reflections and propagation phases between spatially separated coupling points. For real devices, structured residual ripple is a signal to extend the feedline model to a proper cascaded scattering/ABCD network.

The Eq. C11 transcription here is intended as an experimental fitting model and should be checked against the precise conventions and calibration plane of a particular device before using extracted parameters quantitatively.
