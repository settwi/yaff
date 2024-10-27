import astropy.time
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from yaff.fitting import BayesFitter

def plot_data_model(fit: BayesFitter, background_counts: np.ndarray=None):
    fig, ax = plt.subplots(layout='constrained')
    energy_edges = fit.data.count_energy_edges
    data = fit.data.counts
    err = fit.data.counts_error

    if background_counts is None:
        background_counts = np.zeros_like(data)

    # Evaluate the model with the most recently
    # updated set of parameters
    model = fit.eval_model()

    ax.stairs(
        data, energy_edges,
        label='data', color='black', lw=3
    )
    ax.stairs(
        model + background_counts, energy_edges,
        label='model + bkg', color='orange', lw=3
    )

    ax.set(
        title='Model vs Data',
        xlabel='Energy (keV)',
        ylabel='Counts',
        xscale='log',
        yscale='log'
    )
    ax.legend()

    return fig, ax


def stairs_with_error(
        bins: u.Quantity | astropy.time.Time,
        rate: u.Quantity,
        error: u.Quantity=None,
        ax=None,
        label: str=None,
        line_kw: dict=None,
        err_kw: dict=None
):
    '''
    Plot some data (stairs) with nice shaded error bars around the data.

    Styles may be adjusted with the `line_kw` and `err_kw` arguments,
    which get passed to `stairs` and `fill_between`, respectively.
    '''
    ax = ax or plt.gca()
    try:
        ve = ValueError("Rate unit is not the same as the error unit.")
        if error is not None and rate.unit != error.unit:
            raise ve
    except AttributeError:
        raise ve

    try:
        edges, bin_unit = bins.value, bins.unit
    except AttributeError:
        edges = bins.datetime
        bin_unit = ''

    rate, rate_unit = rate.value, rate.unit
    bins = np.unique(edges).flatten()

    st = ax.stairs(rate, bins, label=label, **(line_kw or dict()))

    ax.set(xlabel=bin_unit, ylabel=rate_unit)
    if error is not None:
        col = list(st.get_edgecolor())
        col[-1] = 0.3

        e = error.value
        plot_error = np.concatenate((e, [e[-1]]))
        plot_rate = np.concatenate((rate, [rate[-1]]))
        minus = plot_rate - plot_error
        plus = plot_rate + plot_error
        stacked = np.array((minus, plus))
        minus = stacked.min(axis=0)
        plus = stacked.max(axis=0)
        ax.fill_between(
            x=bins,
            y1=minus,
            y2=plus,
            facecolor=col,
            edgecolor='None',
            step='post',
            **(err_kw or dict())
        )
    return ax
