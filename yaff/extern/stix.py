from astropy.io import fits
import astropy.table as atab
import astropy.time as atime
import astropy.units as u
import numpy as np


def load_pixel_data_to_spectrogram(fn: str) -> dict:
    """Read in the STIX pixel file and sum out
    the detector and pixel axes to get just an
    energy spectrogram.

    Generally, this is preferable over using the
    "spectrogram" data product because the compression
    error is significantly smaller on the pixel count bins.
    """
    pdat = load_pixel_data(fn)

    # sum out the detectors and pixels (axes 1 and 2)
    cts = pdat["counts"].sum(axis=(1, 2))
    cts_err = np.sqrt((pdat["counts_error"] ** 2).sum(axis=(1, 2)))

    pdat.update({"counts": cts, "counts_error": cts_err})
    return pdat


def load_pixel_data(fn: str) -> dict:
    """Read in a STIX pixel data file from the
    data center and convert it to something
    usable with units.
    """
    with fits.open(fn) as f:
        start_date = atime.Time(f["primary"].header["date-beg"])
        earth_timedelta = f["primary"].header["ear_tdel"] << u.s

        data_tab = atab.QTable.read(f, format="fits", hdu="data")

        time_bins = start_date + data_tab["time"]
        dt = data_tab["timedel"]
        time_bins = time_bins - dt / 2
        time_bins = atime.Time(np.concatenate((time_bins, [time_bins[-1] + dt[-1]])))

        # index ordering: (time, detector, pixel, energy bin)
        counts = data_tab["counts"]
        counts_cmp_err = data_tab["counts_comp_err"]
        counts_err = (
            np.sqrt(counts.to_value(u.ct) + counts_cmp_err.to_value(u.ct) ** 2) << u.ct
        )

        triggers = data_tab["triggers"]
        trig_cmp_err = data_tab["triggers_comp_err"]
        trig_cmp_err.shape
        triggers_err = np.sqrt(triggers + trig_cmp_err**2)

        # Estimate the live time and its error using counts, triggers, and
        # circuitry timescales. Function ripped from stixdcpy
        nom_lt = livetime_correct(triggers, counts, dt.to_value(u.s))
        up_lt = livetime_correct(triggers + triggers_err, counts, dt.to_value(u.s))
        lo_lt = livetime_correct(triggers - triggers_err, counts, dt.to_value(u.s))

        # The proportional live time error is approximated as such and
        # propagated through the counts (which are livetime-adjusted)
        prop_live_err = (
            np.abs(up_lt["live_ratio"] - lo_lt["live_ratio"]) / 2 / nom_lt["live_ratio"]
        )

        lt_corr_counts = counts / nom_lt["live_ratio"][:, :, None, None]
        lt_corr_counts_err = np.sqrt(
            counts_err**2 + (prop_live_err[:, :, None, None] * counts) ** 2
        )

        energy_tab = atab.QTable.read(f, format="fits", hdu="energies")
        ebins = np.unique(
            np.column_stack((energy_tab["e_low"], energy_tab["e_high"])).flatten()
        )

        # Detectors 9 and 10 are for coarse flare locating
        # and "background," so we shouldn't do analysis with them
        AVOID_DETECTORS = [9, 10]
        AVOID_MASK = np.array([i not in AVOID_DETECTORS for i in range(1, 33)])
        detector_mask = (data_tab["detector_masks"][0] & AVOID_MASK).astype(bool)

    return {
        "energy_bin_edges": ebins,
        "time_bin_edges": time_bins,
        "livetime": nom_lt["live_ratio"],
        "counts": lt_corr_counts[:, detector_mask, :, :],
        "counts_error": lt_corr_counts_err[:, detector_mask, :, :],
        "earth_spacecraft_dt": earth_timedelta,
    }


def load_srm(fn: str, att_state: str = "unattenuated") -> dict:
    """Read in an IDL-generated SRM file for STIX
    into just a dict.

    The `att_state` specifies the attenuation state
    of the instrument at the time you'd like to fit.
    Possibilities:
     - "unattenuated"
     - "attenuated"
    """
    try:
        # Depending on the attenuation state, the actual response
        # matrix will appear at a different index in the .FITS file
        att_index_map = {"unattenuated": 1, "attenuated": 4}
        srm_idx = att_index_map[att_state]
    except KeyError:
        raise ValueError(
            f"{att_state} is not a valid attenuator choice (choose from attenuated, unattenuated)"
        )

    with fits.open(fn) as f:
        try:
            srm_dat = atab.QTable.read(f[srm_idx], format="fits")
        except IndexError as e:
            raise ValueError(
                "The SRM .FITS file does not have the requested attenuation state"
            ) from e

        ct_energy_dat = atab.QTable.read(f["ebounds"], format="fits")
        area = f[srm_idx].header["geoarea"] << u.cm**2

        # The 'per keV' in this case is count delta E, not photon
        srm = np.array(srm_dat["MATRIX"]) << (u.ct / u.keV / u.ph)
        model_edges = np.unique(
            np.column_stack((srm_dat["ENERG_LO"], srm_dat["ENERG_HI"]))
        ).flatten()
        count_edges = np.unique(
            np.column_stack((ct_energy_dat["E_MIN"], ct_energy_dat["E_MAX"]))
        ).flatten()

        # Remove the horrendous "delta E" from the SRM and
        # multiply in the area
        srm *= np.diff(count_edges)

    return {
        "srm": srm.to(u.ct / u.ph),
        "area": area,
        "photon_energy_edges": model_edges,
        "count_energy_edges": count_edges,
    }


# Taken directly from stixdcpy
# Keep it not a dependency (no bugless stable version as of yet)
def livetime_correct(triggers, counts_arr, time_bins):
    """Live time correction
    Args
        triggers: ndarray
            triggers in the spectrogram
        counts_arr:ndarray
            counts in the spectrogram
        time_bins: ndarray
            time_bins in the spectrogram
    Returns
    live_time_ratio: ndarray
        live time ratio of detectors
    count_rate:
        corrected count rate
    photons_in:
        rate of photons illuminating the detector group
    """
    BETA = 0.94
    FPGA_TAU = 10.1e-6
    ASIC_TAU = 2.63e-6
    TRIG_TAU = FPGA_TAU + ASIC_TAU

    DET_ID_TO_TRIG_INDEX = {
        0: 0,
        1: 0,
        2: 7,
        3: 7,
        4: 2,
        5: 1,
        6: 1,
        7: 6,
        8: 6,
        9: 5,
        10: 2,
        11: 3,
        12: 3,
        13: 4,
        14: 4,
        15: 5,
        16: 13,
        17: 12,
        18: 12,
        19: 11,
        20: 11,
        21: 10,
        22: 13,
        23: 14,
        24: 14,
        25: 9,
        26: 9,
        27: 10,
        28: 15,
        29: 15,
        30: 8,
        31: 8,
    }

    time_bins = time_bins[:, None]
    photons_in = triggers / (time_bins - TRIG_TAU * triggers)
    # photon rate calculated using triggers
    live_ratio = np.zeros((time_bins.size, 32))
    time_bins = time_bins[:, :, None, None]
    count_rate = counts_arr / time_bins
    # print(counts_arr.shape)
    for det in range(32):
        trig_idx = DET_ID_TO_TRIG_INDEX[det]
        nin = photons_in[:, trig_idx]
        live_ratio[:, det] = np.exp(-BETA * nin * ASIC_TAU * 1e-6) / (
            1 + nin * TRIG_TAU
        )
    corrected_rate = count_rate / live_ratio[:, :, None, None]
    return {
        "corrected_rates": corrected_rate,
        "count_rate": count_rate,
        "photons_in": photons_in,
        "live_ratio": live_ratio,
    }
