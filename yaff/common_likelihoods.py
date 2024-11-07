"""Common likelihoods used when processing X-ray data"""

import scipy.stats as st
import numpy as np
from numpy.typing import ArrayLike
from typing import Callable

from yaff import fitting


def poisson_factory(
    restriction: np.ndarray[bool],
) -> Callable[[ArrayLike, ArrayLike], float]:
    """
    Construct a Poisson log likelihood function which restricts evaluation
    using the given `restrict` variable.
    The `restrict` variable should be the same shape as the counts you plan to fit.
    This can be used to, e.g., change the energy range you wish to fit,
    or exclude known bad bins.

    **For those familiar with "cstat" or the "cash" statistic: just use this.**
    It's the exact version of those two.

    The background data is assumed to be a constant with no error for evaluation
    of this likelihood. If background error important, a Negative Binomial likelihood
    should be used, in the sense that it is a Poisson distribution with "extra" variance.
    Wikipedia describes how to map one to the other:
    https://en.wikipedia.org/wiki/Negative_binomial_distribution#Poisson_distribution
    The background error can be added on to the Negative Binomial's
    extra variance.
    """

    def log_likelihood(data: fitting.DataPacket, model: np.ndarray):
        # Set energy bounds to restrict where we care about the likelihood
        # For Poisson likelihood, the model must comprise
        # of integers, otherwise the `logpmf` is sad
        discrete_model = model.astype(int)

        # Any zero-count bins cannot contribute to the log-likelihood for two reasons:
        # 1. a "Poisson distribution" with expected value zero has variance zero, so
        #    pmf(x) = (1 if x == 0 else 0),
        #    meaning ANY model value other than zero will screw up the log likelihood
        # 2. even if the model IS exactly zero, it doesn't affect the log likelihood as
        #    ln(p(0)) = ln(1) = 0.
        restrict = (data.counts > 0) & restriction
        rv = st.poisson(data.counts)
        return rv.logpmf(discrete_model + data.background_counts)[restrict].sum()

    return log_likelihood


def chi_squared_factory(
    restriction: np.ndarray[bool],
) -> Callable[[ArrayLike, ArrayLike], float]:
    """Construct a chi2 likelihood that is weighted by
    errors on the background counts and counts.

    The `restriction` array may be used to restrict the energy
    range which is considered when fitting.
    """

    def chi2_likelihood(data: fitting.DataPacket, model: np.ndarray):
        total_sq_err = data.counts_error**2 + data.background_counts_error**2
        numerator = (data.counts - data.background_counts - model) ** 2

        # Some count bins might be negative, or have zero error,
        # so use nan_to_num
        return -np.nan_to_num((numerator / total_sq_err)[restriction]).sum()

    return chi2_likelihood
