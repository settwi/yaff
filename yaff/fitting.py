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
import scipy.optimize as opt
import scipy.stats as st


def verify(b: bool, s: str, etype: type=ValueError) -> None:
    '''Verify the condition `b`;
       if false, raise an error of type `etype` given `s`'''
    if not b:
        raise etype(s)


def simple_bounds(a: float, b: float):
    '''Wrap a `scipy.stats.uniform` log-PDF to allow easy
       "bounds" making for the non-Bayes inclined.'''
    return st.uniform(loc=a, scale=b-a).logpdf


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
        response_matrix: u.cm**2 * u.ct / u.ph #type: ignore
    ):
        '''All of the basic data needed to
           perform X-ray spectroscopy.

           Performs some basic checks to assert that things
           are the correct shape before allowing the caller
           to proceed.'''
        self.counts: np.ndarray = counts.to_value(u.ct)
        self.counts_error: np.ndarray = counts_error.to_value(u.ct)

        self.background_counts: np.ndarray = background_counts.to_value(u.ct)
        self.background_counts_error: np.ndarray = background_counts_error.to_value(u.ct)

        self.effective_exposure: np.ndarray = np.array(effective_exposure.to_value(u.s))

        self.count_energy_edges: np.ndarray = count_energy_edges.to_value(u.keV)
        self.photon_energy_edges: np.ndarray = photon_energy_edges.to_value(u.keV)

        self.response_matrix = response_matrix.to_value(u.cm**2 * u.ct / u.ph)
        self._verify_dimensions()

    def _verify_dimensions(self):
        # Count data shape verification
        verify(
            self.counts.ndim == 1, "Counts should be 1D.")
        verify(
            self.counts.shape == self.counts_error.shape ==
            self.background_counts.shape ==
            self.background_counts_error.shape,
            "Counts and errors must be the same length.")
        verify(
            self.count_energy_edges.ndim == 1 and
            self.counts.size == self.count_energy_edges.size - 1,
            "Count energy edges must be 1D and have one element more than"
            "the counts array.")
        verify(
            self.counts.size == self.response_matrix.shape[0],
            "Response matrix counts axis (axis 0) size must match count length. "
            "Do you need to take the transpose?")

        # Photon model shape verification
        verify(
            self.photon_energy_edges.ndim == 1, "Photon energy edges should be 1D")
        verify(
            self.photon_energy_edges.size - 1 == self.response_matrix.shape[1],
            "Photon axis of response matrix (axis 1) needs to be one "
            "element shorter than the photon energy edges.")

        # Effective exposure . . . eh
        verify(
            np.ndim(self.effective_exposure) == 0 or
            self.effective_exposure.shape[0] == self.counts.size,
            "Effective exposure must be a scalar, or match the counts shape"
        )

    @property
    def photon_energy_edges(self):
        return self._photon_energy_edges

    @photon_energy_edges.setter
    def photon_energy_edges(self, new):
        self._photon_energy_edges = new
        self.photon_de = np.diff(new)
        self.photon_energy_mids = new[:-1] + self.photon_de/2

    @property
    def count_energy_edges(self):
        return self._count_energy_edges

    @count_energy_edges.setter
    def count_energy_edges(self, new):
        self._count_energy_edges = new
        self.count_de = np.diff(new)
        self.count_energy_mids = new[:-1] + self.count_de/2


class Parameter:
    def __init__(self, quant: u.Quantity, frozen: bool):
        '''A parameter is basically a Quantity with a "frozen" property.'''
        self.value = quant.value
        self.unit = quant.unit
        self.frozen = frozen

    def __repr__(self):
        return f'Parameter({self.value:.2e}, {self.unit}, {self.frozen})'

    def as_quantity(self):
        return self.value << self.unit


class FitsEmcee(abc.ABC):
    '''Mixin indicating that a class
       handles some aspects of the fitting process.
       
       Not to state the obvious, but:
       any method/property that raises NotImplementedError
       needs to... get implemented.
    '''
    def log_posterior(self, pvec, **kwargs) -> float:
        raise NotImplementedError

    @property
    def free_param_vector(self):
        raise NotImplementedError
    
    @property
    def free_param_names(self):
        raise NotImplementedError

    @property
    def num_free_params(self):
        raise NotImplementedError

    @property
    def frozen_parameters(self):
        raise NotImplementedError

    @property
    def flat_named_free_parameters(self) -> tuple[list[str], list[Parameter]]:
        raise NotImplementedError

    def run_emcee(self, emcee_constructor_kw: dict, emcee_run_kw: dict) -> None:
        '''Perform spectroscopy using the data and model given
           in the constructor.

           If you want, things can get updated between fits by modifying
           the object state.'''
        nwalkers = emcee_constructor_kw.pop('nwalkers', os.cpu_count())
        ndim = self.num_free_params

        # Shape the initial state to the number of walkers
        initial_state = self.free_param_vector
        rng = np.random.default_rng()
        perturb = rng.uniform(
            -np.abs(initial_state/10),
            np.abs(initial_state/10),
            size=(nwalkers, ndim)
        )
        initial_state = initial_state + perturb

        with mp.Pool(nwalkers) as p:
            self.emcee_sampler = emcee.EnsembleSampler(
                nwalkers=nwalkers,
                ndim=ndim,
                log_prob_fn=self.log_posterior,
                pool=p,
                kwargs={
                    # Used for mapping back to the params
                    # dict in the object
                    'free_params': self.free_param_names,
                    'frozen_params': self.frozen_parameters
                },
                **emcee_constructor_kw
            )

            emcee_run_kw.setdefault('nsteps', 1000)
            emcee_run_kw.setdefault('progress', True)
            self.emcee_sampler.run_mcmc(
                initial_state,
                **emcee_run_kw
            )

    def emplace_free_parameters(self, vals: list[float]) -> None:
        '''Given a list new free parameters,
           update the internal parameter set.
        '''
        raise NotImplementedError

    def emplace_best_mcmc(self) -> None:
        '''Put the best fit parameters back into the
           parameters.
        '''
        raise NotImplementedError


class BayesFitter(FitsEmcee):
    def __init__(
        self,
        data: DataPacket,
        model_function: Callable[[dict], np.ndarray],
        parameters: dict[str, Parameter],
        log_priors: dict[str, Callable[[float], float]],
        log_likelihood: Callable[..., float]
    ):
        '''Assemble all the pieces we need to
           start doing spectroscopy via Bayesian inference.
           
           Things to note:
           - The log_likelihood should accept:
               - a dict of params
               - a `DataPacket`
               - an evaluated count model, in counts.
           - Each log_prior should accept:
               - A parameter as a float,
                 in the "expected" units,
                 which must be defined by the user.
           - The model_function is expected to return units of ph / keV / cm2 / s.
        '''
        self.data = data
        self.model_function = model_function
        self.parameters = OrderedDict(copy.deepcopy(parameters))
        self.log_priors = OrderedDict(copy.deepcopy(log_priors))
        self.log_likelihood = log_likelihood

        # Gets set later
        self.emcee_sampler = None

    def log_posterior(self, pvec, **kwargs) -> float:
        '''In the Bayes context, the log posterior
           is proportional to the sum of the log priors
           and the log likelihood.
           So, find that sum.

           `pvec` is the "parameter vector" we expect from the
           emcee EnsembleSampler.
           The `**kwargs` contain info on which params are varying.

            We also need to be careful not to use the "self.parameters"
            directly to allow the walkers to vary independently
            in each process.
        '''
        # First, update the `self.parameters` dictionary with the
        # given parameter vector from emcee
        free_params = {
            k: Parameter(v << self.parameters[k].unit, False)
            for (k, v)
            in zip(kwargs.get('free_params', self.free_param_names), pvec)
        }

        # Put all params in one dict; default to the computed frozen ones unless supplied
        all_params = free_params | kwargs.get('frozen_params', self.frozen_parameters)

        ret = 0
        for (n, p) in self.log_priors.items():
            ret += p(all_params[n].value)

        # If we go out of bounds, short-circuit
        # return a bad value
        if np.isneginf(ret) or np.isnan(ret):
            return ret

        model = self.eval_model(params=all_params)
        ret += self.log_likelihood(self.data, model) 
        return ret

    def eval_model(self, params: dict[str, Parameter]=None) -> np.ndarray:
        '''Evaluate the provided photon model and turn it into a counts model'''
        params = params or self.parameters
        photon_model = self.model_function(dict(
            photon_energy_edges=self.data.photon_energy_edges,
            parameters=params
        ))
        count_rate = self.data.response_matrix @ (photon_model * self.data.photon_de)
        counts = count_rate * self.data.effective_exposure
        return counts

    def emplace_best_mcmc(self):
        '''Take the current best (mean) MCMC
           parameter values and assign them to
           the free parameters.
        '''
        if self.emcee_sampler is None:
            raise AttributeError("Emcee has not run yet")

        flat = self.emcee_sampler.flatchain
        best = np.mean(flat, axis=0)
        for (n, best_val) in zip(self.free_param_names, best):
            self.parameters[n].value = best_val

    @property
    def free_parameters(self) -> dict[str, Parameter]:
        return {
            k: copy.deepcopy(v)
            for (k, v) in self.parameters.items()
            if not v.frozen
        }

    @property
    def free_param_vector(self) -> np.ndarray:
        return np.array(list(
            v.value for v in self.parameters.values()
            if not v.frozen
        ))

    @property
    def free_param_names(self) -> list[str]:
        return list(
            k for k in self.parameters.keys()
            if not self.parameters[k].frozen
        )

    @property
    def num_free_params(self) -> int:
        return sum((not p.frozen) for p in self.parameters.values())

    @property
    def frozen_parameters(self) -> dict[str, Parameter]:
        return {
            k: copy.deepcopy(v)
            for (k, v) in self.parameters.items()
            if v.frozen
        }

    def save(self, output_path: str, open_func=open) -> None:
        with open_func(output_path, 'wb') as f:
            dill.dump(self, f)


class CompositeBayesFitter(FitsEmcee):
    '''
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
    of another, just supply a separate, slightly modified, model function.

    The individual BayesFitters don't care about what parameters they receive.
    The CompositeBayesFitter doesn't care about how they evaluate their likelihood.
    Its only job is to coordinate the sharing of parameters between models.

    Once fitting is completed, the models may be decoupled (with parameters properly fixed)
    via the `decouple_models` method.
    It 'squishes' the shared parameters back into the individual models so they may be plotted
    separately or whatever.
    '''
    def __init__(self, individual_fitters: list[BayesFitter], shared_param_names: list[str]):
        '''
        Fit multiple spectra at once, optionally sharing parameters between them.

        `shared_param_names` should be a set of parameter names shared between the
        BayesFitter objects.
        The parameters must have the same (or at least compatible) units.
        '''
        # We don't want to use a `set` because order is not guaranteed
        if len(set(shared_param_names)) != len(shared_param_names):
            raise ValueError("Shared parameters must be unique")

        # Copy the fitters so we don't modify the original objects
        self.fitters = copy.deepcopy(individual_fitters)
        self.shared_params: OrderedDict[str, Parameter] = OrderedDict()
        self._setup_shared(shared_param_names)

    def _setup_shared(self, shared_names: set[str]):
        '''Pop out the shared parameters from the sub-fitters
           and leave them separate for fitting.'''
        for n in shared_names:
            collected = list()
            for (i, fitter) in enumerate(self.fitters):
                try:
                    collected.append(fitter.parameters.pop(n))
                except KeyError:
                    raise ValueError(
                        f"Parameter '{n}' not present in fitter {i} parameter set, "
                        "so it cannot be shared between fitters."
                    )
            
            # Verify units are convertible
            try:
                u.Quantity([c.as_quantity() for c in collected])
            except u.UnitConversionError:
                raise ValueError(f"Not all parameters {n} have compatible units")

            self.shared_params[n] = copy.deepcopy(collected[0])

    def log_posterior(self, pvec, **kwargs):
        '''The log posterior of the CompositeBayesFitter
           is just an appropriate summation of the individual
           BayesFitter log posteriors, supplemented with the shared parameters
           managed by the Composite fitter.
        '''
        # Extract the shared parameters before proceeding
        free_shared_names, free_shared_vals = list(), list()
        for (n, v) in zip(self.free_shared_param_names, pvec):
            free_shared_names.append(n)
            free_shared_vals.append(v)

        # Slice the shared data out of the common args
        num_skip = len(free_shared_names)
        pvec = pvec[num_skip:]
        free_params = kwargs.get('free_params', self.free_param_names)
        remaining_names = free_params[num_skip:]

        # Give slices of the parameter vector to the sub-models
        ret = 0
        for fitter in self.fitters:
            num_free = fitter.num_free_params
            current, pvec = pvec[:num_free], pvec[num_free:]
            
            # Give the fitter its parameters back for units...
            orig = copy.deepcopy(fitter.parameters)
            fitter.parameters.update(self.shared_params)
            ret += fitter.log_posterior(
                np.array(free_shared_vals + list(current)),
                free_params=(free_shared_names + remaining_names[:num_free])
            )
            # and then take them away
            fitter.parameters = orig

        return ret

    def decouple_models(self) -> list[BayesFitter]:
        '''Squish the shared parameters back into the individual models
           and make them individually plottable
        '''
        return NotImplemented

    def emplace_free_parameters(self, vals: list[float]) -> None:
        '''We don't care about names here because the
           parameters are already sorted by (shared --> fitter0 --> fitter1 --> ...)'''
        for (p, v) in zip(self.flat_free_parameters, vals):
            p.value = v

    @property
    def free_param_vector(self):
        '''The free parameters are ordered as follows (assumes N fitters):
            - Zero or more shared params first
            - Non-shared fitter 0 params
            - Non-shared fitter 1 params
            ...
            - Non-shared fitter N params
        '''
        ret = [
            v.value for v in self.shared_params.values()
            if not v.frozen
        ]
        for fitter in self.fitters:
            ret += list(fitter.free_param_vector)
        return np.array(ret)
    
    @property
    def free_param_names(self):
        '''The free parameter names are ordered as follows (assumes N fitters):
            - Zero or more shared names first
            - Non-shared fitter 0 names
            - Non-shared fitter 1 names
            ...
            - Non-shared fitter N names
        '''
        ret = self.free_shared_param_names
        for fitter in self.fitters:
            ret += fitter.free_param_names
        return ret

    @property
    def frozen_parameters(self) -> OrderedDict[str, Parameter]:
        ret = OrderedDict([
            (k, v)
            for (k, v) in self.shared_params.items()
            if v.frozen
        ])
        for fitter in self.fitters:
            ret |= fitter.frozen_parameters
        return ret

    @property
    def num_free_params(self):
        return len(self.free_param_names)

    @property
    def free_shared_param_names(self):
        return list(
            k for k in
            self.shared_params.keys()
            if not self.shared_params[k].frozen
        )

    @property
    def parameters(self) -> dict[str, OrderedDict[str, Parameter]]:
        ret = {
            'shared': self.shared_params
        }
        for (i, fitter) in enumerate(self.fitters):
            ret[f'fitter {i}'] = fitter.parameters
        return ret

    @property
    def flat_free_parameters(self) -> list[Parameter]:
        '''Extract the Parameter objects from the shared param
           dictionary and the sub-fitter objects'''
        ret = list(p for p in self.shared_params.values() if not p.frozen)
        for f in self.fitters:
            ret += list(p for p in f.parameters if not p.frozen)
        return ret

    @property
    def flat_named_free_parameters(self) -> tuple[list[str], list[Parameter]]:
        '''Couple flattened names with flattened parameters'''
        return (
            list(self.free_param_names),
            list(self.flat_free_parameters)
        )
            

def normal_minimize(fit_obj: FitsEmcee, **minimize_kw) -> list[str, float]:
    '''Minimize the log posterior provided by the fit_obj
       update the object's parameters with the "best fit" ones.
       
       Intended to give a decent starting guess to the
       MCMC process.
       
       The `**minimize_kw` are passed directly to
       `scipy.optimize.minimize`.

       This function could be implemented as a mixin class,
       or something like that.
       But, eh.
    '''
    # We provide the function in this case,
    # so if the user specified one, get rid of it
    if 'fun' in minimize_kw:
        raise ValueError("Do not specify the objective function")

    kw = dict(x0=fit_obj.free_param_vector, method='Nelder-Mead')
    kw.update(minimize_kw)

    # Save the names of the free parameters before minimizing
    free_names = fit_obj.free_param_names
    frozen = fit_obj.frozen_parameters
    def objective_function(param_vector, *_) -> float:
        # We're using a minimization algorithm but we want the
        # log posterior to be maximized.
        # So, invert the sign and call it good
        return -fit_obj.log_posterior(
            param_vector, free_params=free_names, frozen_params=frozen)

    res = opt.minimize(
        fun=objective_function,
        **kw
    )

    if not res.success:
        warnings.warn('minimization was not successful')

    # Put the best parameters back into the fitting object
    fit_obj.emplace_free_parameters(res.x)
    
    return fit_obj
