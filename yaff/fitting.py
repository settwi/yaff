import abc
from collections import OrderedDict
import copy
import os
from typing import Callable
import warnings

import astropy.units as u
import dill
import emcee

# NB: not `multiprocessing`, we're using `multiprocess`
# which uses `dill` instead of `pickle` to save stuff
import multiprocess as mp
import numpy as np
from numpy.typing import ArrayLike
import scipy.optimize as opt
import scipy.stats as st
from . import rebin_flux


class DataPacket:
    @u.quantity_input
    def __init__(
        self,
        counts: u.ct,
        counts_error: u.ct,
        background_counts: u.ct,
        background_counts_error: u.ct,
        effective_exposure: u.s,
        count_energy_edges: u.keV,
        photon_energy_edges: u.keV,
        response_matrix: u.cm**2 * u.ct / u.ph,  # type: ignore
    ):
        """All of the basic data needed to
        perform X-ray spectroscopy.

        Performs some basic checks to assert that things
        are the correct shape before allowing the caller
        to proceed."""
        self.counts: np.ndarray = counts.to_value(u.ct)
        self.counts_error: np.ndarray = counts_error.to_value(u.ct)

        self.background_counts: np.ndarray = background_counts.to_value(u.ct)
        self.background_counts_error: np.ndarray = background_counts_error.to_value(
            u.ct
        )

        self.effective_exposure: np.ndarray = np.array(effective_exposure.to_value(u.s))

        self.count_energy_edges: np.ndarray = count_energy_edges.to_value(u.keV)
        self.photon_energy_edges: np.ndarray = photon_energy_edges.to_value(u.keV)

        self.response_matrix = response_matrix.to_value(u.cm**2 * u.ct / u.ph)
        self._verify_dimensions()

    def _verify_dimensions(self):
        # Count data shape verification
        verify(self.counts.ndim == 1, "Counts should be 1D.")
        verify(
            self.counts.shape
            == self.counts_error.shape
            == self.background_counts.shape
            == self.background_counts_error.shape,
            "Counts and errors must be the same length.",
        )
        verify(
            self.count_energy_edges.ndim == 1
            and self.counts.size == self.count_energy_edges.size - 1,
            "Count energy edges must be 1D and have one element more than"
            "the counts array.",
        )
        verify(
            self.counts.size == self.response_matrix.shape[0],
            "Response matrix counts axis (axis 0) size must match count length. "
            "Do you need to take the transpose?",
        )

        # Photon model shape verification
        verify(self.photon_energy_edges.ndim == 1, "Photon energy edges should be 1D")
        verify(
            self.photon_energy_edges.size - 1 == self.response_matrix.shape[1],
            "Photon axis of response matrix (axis 1) needs to be one "
            "element shorter than the photon energy edges.",
        )

        # Effective exposure . . . eh
        verify(
            np.ndim(self.effective_exposure) == 0
            or self.effective_exposure.shape[0] == self.counts.size,
            "Effective exposure must be a scalar, or match the counts shape",
        )

        if self.response_matrix.shape[0] == self.response_matrix.shape[1]:
            warnings.warn(
                "\nYour response matrix is square."
                "\nMake sure it is oriented properly, C = (SRM @ P)."
                "\nCan't tell from photon vs count edge shapes"
            )

    @property
    def photon_energy_edges(self):
        return self._photon_energy_edges

    @photon_energy_edges.setter
    def photon_energy_edges(self, new):
        self._photon_energy_edges = new
        self.photon_de = np.diff(new)
        self.photon_energy_mids = new[:-1] + self.photon_de / 2

    @property
    def count_energy_edges(self):
        return self._count_energy_edges

    @count_energy_edges.setter
    def count_energy_edges(self, new):
        self._count_energy_edges = new
        self.count_de = np.diff(new)
        self.count_energy_mids = new[:-1] + self.count_de / 2


class Parameter:
    def __init__(self, quant: u.Quantity, frozen: bool):
        """A parameter is basically a Quantity with a "frozen" property."""
        self.value = quant.value
        self.unit = quant.unit
        self.frozen = frozen

    def __repr__(self) -> str:
        return f"Parameter[{self.value:.2e}, {self.unit}, frozen={self.frozen}]"

    def as_quantity(self) -> u.Quantity:
        return self.value << self.unit


class FitsEmceeMixin(abc.ABC):
    """Mixin indicating that a class
    handles some aspects of the fitting process,
    using `emcee.EnsembleSampler`.

    Not to state the obvious, but:
    any method/property that raises NotImplementedError
    needs to... get implemented.
    """

    def log_posterior(self, pvec: ArrayLike, **kwargs) -> float:
        raise NotImplementedError

    @property
    def free_param_vector(self) -> list[float]:
        raise NotImplementedError

    @property
    def num_free_params(self) -> int:
        raise NotImplementedError

    def run_emcee(self, emcee_constructor_kw: dict, emcee_run_kw: dict) -> None:
        """Perform spectroscopy using the data and model given
        in the constructor.

        If you want, things can get updated between fits by modifying
        the object state."""
        nwalkers = emcee_constructor_kw.pop("nwalkers", os.cpu_count())
        ndim = self.num_free_params

        # Shape the initial state to the number of walkers
        initial_state = np.array(self.free_param_vector)
        rng = np.random.default_rng()
        perturb = rng.uniform(
            -np.abs(initial_state / 10),
            np.abs(initial_state / 10),
            size=(nwalkers, ndim),
        )
        initial_state = initial_state + perturb

        with mp.Pool(nwalkers) as p:
            self.emcee_sampler = emcee.EnsembleSampler(
                nwalkers=nwalkers,
                ndim=ndim,
                log_prob_fn=self.log_posterior,
                pool=p,
                **emcee_constructor_kw,
            )

            emcee_run_kw.setdefault("nsteps", 1000)
            emcee_run_kw.setdefault("progress", True)
            self.emcee_sampler.run_mcmc(initial_state, **emcee_run_kw)

    def emplace_free_parameters(self, vals: list[float]) -> None:
        """Given a list new free parameters,
        update the internal parameter set.
        """
        raise NotImplementedError

    def save(self, output_path: str, open_func=open) -> None:
        """Save the object to a pickle using `dill`;
        it can be optionally compressed by supplying an `open_func`"""
        with open_func(output_path, "wb") as f:
            dill.dump(self, f)


class BayesFitter(FitsEmceeMixin):
    def __init__(
        self,
        data: DataPacket,
        model_function: Callable[[dict], np.ndarray],
        parameters: dict[str, Parameter],
        log_priors: dict[str, Callable[[float], float]],
        log_likelihood: Callable[[ArrayLike, ArrayLike], float],
    ):
        """Assemble all the pieces we need to
        start doing spectroscopy via Bayesian inference.

        Things to note:
        - The log_likelihood should accept:
            - a `DataPacket`
            - an evaluated count model, in counts.
        - Each log_prior should accept:
            - A parameter as a float,
              in the "expected" units,
              which must be defined by the user.
        - The model_function is expected to return units of ph / keV / cm2 / s.
        """
        self.data = data
        self.model_function = model_function
        self.parameters = OrderedDict(copy.deepcopy(parameters))

        self.log_priors: OrderedDict[str, Callable[[float], float]] = OrderedDict()
        for k in self.parameters:
            # Loop over the keys so that we maintain
            # the same ordering as the params
            self.log_priors[k] = log_priors[k]
        self.log_likelihood = log_likelihood

        # Gets set later
        self.emcee_sampler: emcee.EnsembleSampler | None = None

    def log_posterior(self, pvec) -> float:
        """In the Bayes context, the log posterior
        is equal to the sum of the log priors
        and the log likelihood, up to an additive constant.

        `pvec` is the "parameter vector" we expect from the
        emcee EnsembleSampler.

        We also need to be careful not to use the "self.parameters"
        directly to allow the walkers to vary independently
        in each process.
        """
        # First, construct this dictionary with the
        # given parameter vector from emcee.
        # We use a copy to keep things happy during multiprocessing
        free_params = {
            k: Parameter(v << self.parameters[k].unit, False)
            for (k, v) in zip(self.free_param_names, pvec)
        }

        # Put all params in one dict
        all_params = free_params | self.frozen_parameters
        ret = 0
        for n, p in self.log_priors.items():
            ret += p(all_params[n].value)

        # If we go out of bounds on any parameter,
        # short-circuit return a bad value
        if np.isneginf(ret) or np.isnan(ret):
            return ret

        model = self.eval_model(params=all_params)
        ret += self.log_likelihood(self.data, model)
        return ret

    def eval_model(self, params: dict[str, Parameter] | None = None) -> np.ndarray:
        """Evaluate the associated photon model and turn it into a counts model"""
        params = params or self.parameters
        photon_model = self.model_function(
            dict(photon_energy_edges=self.data.photon_energy_edges, parameters=params)
        )
        count_rate = self.data.response_matrix @ (photon_model * self.data.photon_de)
        counts = count_rate * self.data.effective_exposure
        return counts

    def emplace_best_mcmc(self) -> None:
        """Take the current best (mean) MCMC
        parameter values and assign them to
        the free parameters.
        """
        if self.emcee_sampler is None:
            raise AttributeError("Emcee has not run yet")
        flat = self.emcee_sampler.flatchain
        self.emplace_free_parameters(np.mean(flat, axis=0))

    def emplace_free_parameters(self, vals: list[float]) -> None:
        """Given a list new free parameters,
        update the internal parameter set.
        """
        for k, v in zip(self.free_param_names, vals):
            self.parameters[k].value = v

    def eval_priors(self) -> float:
        """Evaluate and sum all priors with the
        current parameters"""
        ret = 0
        for k, prior in self.log_priors.items():
            ret += prior(self.parameters[k].value)
        return ret

    def generate_model_samples(self, num: int) -> np.ndarray:
        """Generate model samples from the parameter chains in
        the associated `emcee.EnsembleSampler`.
        If no sampler is present, current parameters
        are used.
        """
        if self.emcee_sampler is None:
            warnings.warn(
                "You haven't run the emcee sampler yet, "
                "so we can only generate one model sample."
            )
            return [self.eval_model()]

        param_samples = np.random.default_rng().choice(
            self.emcee_sampler.flatchain, size=num
        )

        ret = list()
        for param_set in param_samples:
            self.emplace_free_parameters(param_set)
            ret.append(self.eval_model())
        return np.array(ret)

    @property
    def free_parameters(self) -> list[Parameter]:
        return list(
            copy.deepcopy(v)
            for v in self.parameters.values()
            if not v.frozen
        )

    @property
    def free_param_vector(self) -> list[float]:
        return list(v.value for v in self.parameters.values() if not v.frozen)

    @property
    def free_param_names(self) -> list[str]:
        return list(k for k in self.parameters.keys() if not self.parameters[k].frozen)

    @property
    def num_free_params(self) -> int:
        return len(self.free_param_names)

    @property
    def frozen_parameters(self) -> dict[str, Parameter]:
        return {k: copy.deepcopy(v) for (k, v) in self.parameters.items() if v.frozen}


class CompositeBayesFitter(FitsEmceeMixin):
    """
    A `CompositeBayesFitter` accepts:
        - a list of individual `BayesFitter`s, with the data already loaded, and
        - a set of parameter names shared between the individual fitters.

    The fitter log_likelihoods are called individually with a parameter vector which
    has been properly updated by including shared parameter information.

    **If you want the parameters to be tied together in a fancy way**
    besides just being shared:
    you need to add that to the model function given to the fitter whose
    behavior is special.
    For example, if you want a particular spectrum to be fit with 2x the temperature
    of another, just supply a separate, slightly modified, model function which maps
    T --> 2*T.
    """

    def __init__(
        self, individual_fitters: list[BayesFitter], shared_param_names: list[str]
    ):
        """
        Fit multiple spectra at once, optionally sharing parameters between them.

        `shared_param_names` should be a set of parameter names shared between the
        BayesFitter objects.
        The parameters must have the same units--not just compatible ones.
        """
        if len(shared_param_names) == 0:
            raise ValueError(
                "If you have no shared parameters, "
                "fit the spectra individually instead."
            )
        # We don't want to use a `set` because order is not guaranteed,
        # so instead check if we were given a unique ordered array of params.
        if len(set(shared_param_names)) != len(shared_param_names):
            raise ValueError("Shared parameters must be unique")

        # Copy the fitters so we don't modify the original objects
        self.fitters = copy.deepcopy(individual_fitters)
        self.shared_params: OrderedDict[str, Parameter] = OrderedDict()
        self._setup_shared(shared_param_names)

        self.emcee_sampler: emcee.EnsembleSampler | None = None

    def _setup_shared(self, shared_names: set[str]):
        """Pop the shared parameters from the sub-fitters
        and leave them separate for fitting."""
        for n in shared_names:
            collected = list()
            for i, fitter in enumerate(self.fitters):
                try:
                    collected.append(fitter.parameters.pop(n))
                except KeyError:
                    raise ValueError(
                        f"Parameter '{n}' not present in fitter {i} parameter set, "
                        "so it cannot be shared between fitters."
                    )

            # Verify units are consistent: should be the same for value conversion
            if not all(c.unit == collected[0].unit for c in collected):
                raise ValueError(f"All parameters '{n}' must have the same units")

            self.shared_params[n] = copy.deepcopy(collected[0])

    def log_posterior(self, pvec):
        """The log posterior of the CompositeBayesFitter
        is just an appropriate summation of the individual
        BayesFitter log posteriors, supplemented with the shared parameters
        managed by the Composite fitter.
        """
        # Slice the shared data out of the common args
        free_shared_names = self.free_shared_param_names
        num_skip = len(free_shared_names)
        free_shared_vals, pvec = pvec[:num_skip], pvec[num_skip:]

        # Give slices of the parameter vector to the sub-models
        ret = 0
        for fitter in self.fitters:
            # Slice out the current model's free parameters
            # from the provided vector
            num_free = len(fitter.free_param_names)
            current, pvec = pvec[:num_free], pvec[num_free:]

            # Update/set any parameters in the sub-fitter
            orig = copy.deepcopy(fitter.parameters)
            fitter.parameters.update(self.shared_params)

            # Put the shared parameters after the common ones
            # in case some got set when updating the ordered dicts
            all_free = np.concatenate((current, free_shared_vals))
            cur = fitter.log_posterior(all_free)
            ret += cur

            # Remove the shared parameters from the sub-fitter
            # after evaluating its model
            fitter.parameters = orig

        return ret

    def emplace_free_parameters(self, vals: list[float]) -> None:
        """Update all free parameter values from the given list.
        We don't care about names here because the
        parameters are already sorted by (shared --> fitter0 --> fitter1 --> ...)"""
        for p, v in zip(self.free_parameters, vals):
            p.value = v

    def generate_model_samples(self, num: int) -> np.ndarray:
        """Generate model samples from the parameter chains in
        the associated `emcee.EnsembleSampler`.
        If no sampler is present, current parameters
        are used.
        """
        # TODO
        return NotImplemented

    @property
    def free_param_vector(self) -> list[float]:
        """The free parameter values are ordered as follows (assumes N fitters):
        - Zero or more shared params first
        - Non-shared fitter 0 params
        - Non-shared fitter 1 params
        ...
        - Non-shared fitter N params
        """
        ret = [v.value for v in self.shared_params.values() if not v.frozen]
        for fitter in self.fitters:
            free_vec = [
                p.value
                for (k, p) in fitter.parameters.items()
                if (k not in self.shared_params and not p.frozen)
            ]
            ret += free_vec
        return ret

    @property
    def free_param_names(self) -> list[str]:
        """The free parameter names are ordered as follows (assumes N fitters):
        - Zero or more shared names first
        - Non-shared fitter 0 names
        - Non-shared fitter 1 names
        ...
        - Non-shared fitter N names
        """
        ret = self.free_shared_param_names
        for fitter in self.fitters:
            ret += [n for n in fitter.free_param_names if n not in self.shared_params]
        return ret

    @property
    def num_free_params(self) -> int:
        return len(self.free_param_names)

    @property
    def free_shared_param_names(self) -> list[str]:
        return list(
            k for k in self.shared_params.keys() if not self.shared_params[k].frozen
        )

    @property
    def parameters(self) -> dict[str, OrderedDict[str, Parameter]]:
        ret = {"shared": self.shared_params}
        for i, fitter in enumerate(self.fitters):
            ret[f"fitter {i}"] = {
                k: p
                for (k, p) in fitter.parameters.items()
                if k not in self.shared_params
            }
        return ret

    @property
    def free_parameters(self) -> list[Parameter]:
        """Extract the Parameter objects from the shared param
        dictionary and the sub-fitter objects"""
        ret = list(p for p in self.shared_params.values() if not p.frozen)
        for f in self.fitters:
            ret += list(
                copy.deepcopy(p)
                for (k, p) in f.parameters.items()
                if not p.frozen and k not in self.shared_params
            )
        return ret

    @property
    def free_param_units(self) -> list[u.Unit]:
        return list(p.unit for p in self.free_parameters)


class BayesFitterWithGain(BayesFitter):
    def __init__(
        self,
        data: DataPacket,
        model_function: Callable[[dict], np.ndarray],
        parameters: dict[str, Parameter],
        log_priors: dict[str, Callable[[float], float]],
        log_likelihood: Callable[[ArrayLike, ArrayLike], float],
    ):
        super().__init__(data, model_function, parameters, log_priors, log_likelihood)
        # Add gain parameters which can be modified in fitting
        self.parameters["gain_slope"] = Parameter(1.0 << u.one, True)
        self.parameters["gain_offset"] = Parameter(0.0 << u.keV, True)
        self.log_priors["gain_slope"] = simple_bounds(0.5, 1.5)
        self.log_priors["gain_offset"] = simple_bounds(-1, 1)
        warnings.warn("\nGain slope and offset parameters/priors added to the parameter/prior ODicts.")

    def eval_model(self, params=None):
        r"""Evaluate the associated photon model and turn it into a counts model.
        Acts with the gain parameters on the model count flux, interpolating
        it onto the new energy bins via flux-conserving rebinning.

        The gain parameters are defined like in xspec, but do not act
        on the response matrix itself, rather on the count model.

        There is no good way to fit the "gain" to the response. As the XSPEC manual states,
            "**\*CAUTION\*** This command is to be used with extreme care for investigation of the response properties.
            To properly fit data, the response matrix should be recalculated explicitly (outside of XSPEC)
            using any modified gain information derived."

        Interpolating the model flux is significantly simpler than interpolating the effective area.
        So we do that here.
        """
        slope = self.parameters["gain_slope"].value
        intercept = self.parameters["gain_offset"].as_quantity().to_value(u.keV)
        new_edges = (self.data.count_energy_edges / slope) - intercept

        original_counts = super().eval_model(params)
        return rebin_flux.rebin_histogram_cdf(
            self.data.count_energy_edges, original_counts, new_edges
        )


def levenberg_minimize(
    fitter: BayesFitter, restriction: np.ndarray[bool] = None, **scipy_kwargs
) -> BayesFitter:
    """Given a Bayes fitter, minimize its parameters using the Levenberg-Marquadt
    (weighted) least squares minimization like XSPEC does.

    The `restriction` parameter is used to optionally exclude count bins
    which shouldn't be considered when fitting.

    ---

    Levenberg-Marquadt operates on **all** of the model and data
    count bins. So, it tends to be more robust (and converge faster) than
    algorithms which are based on a single summary number.

    Applying this minimizer is a good first step before doing MCMC
    as it will (robustly) put the parameter vector near its global minimum.
    """

    if restriction is None:
        restriction = np.ones_like(fitter.data.counts, dtype=bool)

    def residual_function(vector: ArrayLike):
        fitter.emplace_free_parameters(vector)
        if np.isneginf(p := fitter.eval_priors()) or np.isnan(p):
            # if we go out of bounds, don't evaluate the model
            return np.full(fitter.data.counts.shape, np.inf)

        mod = fitter.eval_model()
        compare = fitter.data.counts - fitter.data.background_counts

        # weight each residual by the error in each bin:
        # larger error = less impact on the fit
        total_error = np.sqrt(
            fitter.data.counts_error**2 + fitter.data.background_counts_error**2
        )

        ret = (mod - compare) / total_error
        ret[~restriction] = 0
        # Any "zero-error" bins need to get deleted
        return np.nan_to_num(ret, copy=False, nan=0, posinf=0, neginf=0)

    scipy_kwargs["method"] = "lm"
    guess = fitter.free_param_vector
    res = opt.least_squares(fun=residual_function, x0=guess, **scipy_kwargs)

    if not res.success:
        warnings.warn("minimization failed; whatever")

    fitter.emplace_free_parameters(res.x)
    return fitter


def verify(b: bool, s: str, etype: type = ValueError) -> None:
    if not b:
        raise etype(s)


def simple_bounds(a: float, b: float):
    """Wrap a `scipy.stats.uniform` log-PDF to allow easy
    "bounds" making for the non-Bayes inclined."""
    return st.uniform(loc=a, scale=b - a).logpdf
