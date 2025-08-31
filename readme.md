# `yaff`: Yet Another F'ing (X-ray) Fitter
## Bayesian inference for X-ray spectroscopy in Python
This package facilitates easy Bayesian inference on X-ray data products.
It is a lightweight implementation based on [`emcee`](https://emcee.readthedocs.io/en/stable/)
which incorporates
[`astropy.units`](https://docs.astropy.org/en/latest/units/index.html).
"Bounds" and other (more generic) parameter prior distributions are implemented via
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html).

**Please see the `examples` folder for some complete examples. (TODO: make them a gallery)**

## Table of contents
- [Installation](#installation)
- [List of examples](#examples)
- [Spectroscopy fundamentals](#x-ray-spectroscopy-basics)
- [Package overview](#package-overview)

## Installation
`yaff` can be installed from GitHub.
It's recommended to use `uv` to manage your Python versions and virtual environments: [uv project GitHub page, with installation instructions](https://github.com/astral-sh/uv).
It is very fast and works well.

Clone the repository and run
```bash
cd /path/to/where/you/cloned/yaff
# Optionally--but preferably--make a venv to work in
# here is an example of how to do that using the `uv` tool
uv venv
source .venv/bin/activate

uv pip install -e .
# or omit the `uv` if you are just using pip

# If you want to install the packages required by the examples as well:
uv pip install -e .[examples]
```

## Examples
All of these can be found in `examples` directory. (TODO: make them a gallery)

## X-ray spectroscopy basics
### Spectroscopy: background
Analyzing X-rays from the Sun and other stars is straightforward
    but distinct from other wavelengths.
For X-ray instruments (operating at energies greater than ~1 keV),
    each photon is counted individually.
The intensity is recorded as "counts/second" rather than, e.g.,
    watts.

Furthermore,
    X-rays are of high enough energy that inelastic photon-matter
    interactions become very important to consider when performing
    any kind of serious energy analysis.
The interactions between matter and the X-ray photons distinguish
    the concept of a **count** from a **photon**.
A photon of a given energy may register as a count of a similar,
    or significantly smaller,
    energy.

The _details_ of the photon-matter interaction is captured in a matrix.
This matrix goes by various different names, depending on who you ask:
- Instrument response matrix (IRM)
- Detector response matrix (DRM)
- Spectral response matrix (SRM)
- ... probably others.

Sometimes these matrices have units of (count / photon).
Sometimes they have other units, like (square centimeter times count / photon / keV).

All of this is confusing.
At the most fundamental level,
    a _response matrix_ converts an "average" photon to an "average" count;
    it is a probability mapping from incident photon energy
    to registered count energy.
**Hereafter:
    the response matrix quantifies the conversion of photons to counts,
    and has units of count / photon.**
The geometric area of a detector may be multipled into this response,
    when appropriate, to modify units.

_As an aside: putting a "per-keV" in the response is a TERRIBLE idea. Photon model energy spacings may be very different from instrument count energy spacings, and this ambiguity can lead to insidious and confusing results._

### Spectroscopy: mechanics
X-ray spectroscopy consists of just a few steps:
1. Compute a model of the photon emission.
2. Convert the photon emission model into a measured "count" model using the response matrix.
3. Compare the count model to your data.
4. Change the model parameters to get a better match.
5. Repeat until you're happy.

That's it. Once you are "happy" in a rigorous sense, you are done performing spectroscopy.

## Package overview
TBD
