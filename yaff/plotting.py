import astropy.time
import astropy.units as u
import corner
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import numpy.typing
from yaff.fitting import BayesFitter, FitsEmceeMixin, Parameter


def plot_data_model(
    fit: BayesFitter,
    model_samples: numpy.typing.ArrayLike | None = None,
    fig: Figure = None,
):
    """Given a BayesFitter,
    plot the data, and on top of the data,
    plot a few model samples.

    `model_samples` should be an `ArrayLike` of models
    evaluated at various parameters.

    If no `model_samples` are provided, the fitter is evaluated
    at its current parameter set for plotting.

    Also visualizes the residual for each model sample.
    """
    fig = fig or plt.figure()
    if model_samples is None:
        model_samples = [fit.eval_model()]

    gskw = {"height_ratios": (4, 1), "hspace": 0.05}
    data_ax, err_ax = fig.subplots(ncols=1, nrows=2, sharex=True, gridspec_kw=gskw)

    energy_edges = fit.data.count_energy_edges
    data = fit.data.counts
    err = fit.data.counts_error
    bkg = fit.data.background_counts
    bkg_err = fit.data.background_counts_error

    # Plot the data
    stairs_with_error(
        energy_edges << u.keV,
        (data - bkg) << u.ct,
        np.sqrt(err**2 + bkg_err**2) << u.ct,
        data_ax,
        label="data - bkg",
        line_kw={"color": "blue"},
    )

    # Plot the background
    stairs_with_error(
        bins=energy_edges << u.keV,
        quant=bkg << u.ct,
        error=bkg_err << u.ct,
        ax=data_ax,
        label="bkg",
        line_kw={"color": "orange"},
    )

    alpha = min(1, 2 / len(model_samples))
    for count_model in model_samples:
        data_ax.stairs(count_model, energy_edges, color="black", alpha=alpha)

        residual = (data - count_model - bkg) / np.sqrt(err**2 + bkg_err**2)
        err_ax.stairs(residual, energy_edges, color="black", alpha=alpha)

    err_ax.axhline(color="magenta", zorder=-1, alpha=0.2)
    data_ax.legend()

    data_ax.set(
        title="Model vs Data", ylabel="Counts", xscale="log", yscale="log", xlabel=None
    )

    err_ax.set(xlabel="Energy (keV)", ylabel="$(D - M) / \\sigma_D$", ylim=(-5, 5))

    return {"fig": fig, "data_ax": data_ax, "error_ax": err_ax}


def plot_parameter_chains(
    fitter: FitsEmceeMixin,
    names: list[str],
    params: list[Parameter],
    fig: Figure = None,
):
    """Given a FitsEmcee-like object, plot the parameter MCMC chains.
    You may optionally provide a figure to plot on."""
    if fitter.emcee_sampler is None:
        raise ValueError("Emcee needs to be run first")

    fig = fig or plt.figure(figsize=(8, 6))
    param_units = list(p.unit for p in params)

    chains = fitter.emcee_sampler.chain
    axes = []
    num_rows = len(names)
    for i in range(num_rows):
        axes.append(ax := fig.add_subplot(num_rows, 1, i + 1))
        name = names[i]
        unit = param_units[i]
        ax.plot(chains[..., i].T, color="black", alpha=0.3)
        ax.set(title=name, ylabel=unit)

    return {"fig": fig, "axes": axes}


def corner_plot(fitter: FitsEmceeMixin, burnin: int, fig: Figure = None):
    """Take a FitsEmcee-like object and plot some parameter corner plots."""
    corner_chain = fitter.emcee_sampler.flatchain[burnin:]
    param_names = fitter.free_param_names

    fig = fig or plt.figure(figsize=(20, 20), layout="tight")
    corner.corner(
        corner_chain,
        fig=fig,
        bins=20,
        labels=param_names,
        quantiles=(0.05, 0.5, 0.95),
        show_titles=True,
    )
    return fig


def stairs_with_error(
    bins: u.Quantity | astropy.time.Time,
    quant: u.Quantity,
    error: u.Quantity = None,
    ax=None,
    label: str = None,
    line_kw: dict = None,
    err_kw: dict = None,
):
    """
    Plot some data (stairs) with nice shaded error bars around the data.

    Styles may be adjusted with the `line_kw` and `err_kw` arguments,
    which get passed to `stairs` and `fill_between`, respectively.
    """
    ax = ax or plt.gca()
    try:
        ve = ValueError("Rate unit is not the same as the error unit.")
        if error is not None and quant.unit != error.unit:
            raise ve
    except AttributeError:
        raise ve

    try:
        edges, bin_unit = bins.value, bins.unit
    except AttributeError:
        edges = bins.datetime
        bin_unit = ""

    quant, quant_unit = quant.value, quant.unit
    bins = np.unique(edges).flatten()

    st = ax.stairs(quant, bins, label=label, **(line_kw or dict()))

    ax.set(xlabel=bin_unit, ylabel=quant_unit)
    if error is not None:
        col = list(st.get_edgecolor())
        col[-1] = 0.3

        e = error.value
        plot_error = np.concatenate((e, [e[-1]]))
        plot_quant = np.concatenate((quant, [quant[-1]]))
        minus = plot_quant - plot_error
        plus = plot_quant + plot_error
        stacked = np.array((minus, plus))
        minus = stacked.min(axis=0)
        plus = stacked.max(axis=0)
        ax.fill_between(
            x=bins,
            y1=minus,
            y2=plus,
            facecolor=col,
            edgecolor="None",
            step="post",
            **(err_kw or dict()),
        )
    return ax
