import astropy.units as u
from sunkit_spex.extern import rhessi
import numpy as np
from yaff import fitting


def adapt_rhessi_data(rl: rhessi.RhessiLoader) -> fitting.DataPacket:
    """Convert the sunkit-spex format into a fitting.DataPacket"""
    # One-D arrays for energy bins
    count_ebins = np.unique(rl["count_channel_bins"].flatten()) << u.keV
    photon_ebins = np.unique(rl["photon_channel_bins"].flatten()) << u.keV

    cts = (
        rl._loaded_spec_data["counts"] << u.ct
    )  # (rl._spectrum['counts'] / rl._spectrum['livetime'])
    err = (
        rl._loaded_spec_data["count_error"] << u.ct
    )  # (rl._spectrum['counts_err'] / rl._spectrum['livetime'])

    bkg_cts = (
        rl["extras"]["background_rate"]
        * rl["count_channel_binning"]
        * rl["effective_exposure"]
    )
    bkg_err = (
        rl["extras"]["background_rate_error"]
        * rl["count_channel_binning"]
        * rl["effective_exposure"]
    )

    effective_exposure = rl._loaded_spec_data["effective_exposure"] << u.s
    return fitting.DataPacket(
        counts=cts << u.ct,
        counts_error=err << u.ct,
        background_counts=bkg_cts << u.ct,
        background_counts_error=bkg_err << u.ct,
        effective_exposure=effective_exposure << u.s,
        count_energy_edges=count_ebins << u.keV,
        photon_energy_edges=photon_ebins << u.keV,
        response_matrix=rl["srm"].T << (u.cm**2 * u.ct / u.ph),
    )
