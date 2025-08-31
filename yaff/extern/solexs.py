from astropy.io import fits
import astropy.time as atime
import astropy.units as u
import numpy as np


@u.quantity_input()
def time_from_header(head: fits.header.Header) -> u.s:
    """Compute the observation time from a FITS header"""
    start, stop = head["TSTART"], head["TSTOP"]
    return (atime.Time(stop) - atime.Time(start)).to(u.s)


def read_counts(data_fn: str, sys_uncert=0.04) -> dict[str, u.Quantity]:
    """Read in the SoLEXS counts data and apply a systematic
    uncertianty, if desired.

    Defaults to 4%, as recommended by the SoLEXS user manual.
    """
    with fits.open(data_fn) as dat:
        cts = np.array(dat[1].data["COUNTS"], dtype=int) << u.ct

        # Assume Poisson error, then add on systematics
        sys_uncert = 0.04
        err = np.sqrt(cts.to_value(u.ct))
        err = np.sqrt(err**2 + (sys_uncert * cts.to_value(u.ct)) ** 2) << u.ct
        exposure = time_from_header(dat[1].header)

    return {"counts": cts, "counts_error": err, "exposure": exposure}


def read_rmf(rmf_fn: str) -> dict[str, u.Quantity]:
    """
    Read in a SoLEXS RMF file.
    The RMF file describes only how photon energies get distributed
    probabilistically into count energy bins (e.g. the energy resolution component).
    It does not contain effective area information.
    """
    with fits.open(rmf_fn) as rd:
        count_ebins = (
            np.concatenate((rd[1].data["E_MIN"], [rd[1].data["E_MAX"][-1]])) << u.keV
        )
        matrix = rd[2].data["MATRIX"]
        return {
            "count_energy_bins": count_ebins << u.keV,
            "redistribution_matrix": matrix.T << (u.ct / u.ph),
        }


def read_arf(arf_fn: str) -> dict[str, u.Quantity]:
    with fits.open(arf_fn) as f:
        dat = f[1].data
        ebins = (
            np.concatenate((dat["energ_lo"], [dat["energ_hi"][-1]]), dtype=float)
            << u.keV
        )
        eff_area = dat["specresp"]
        return {
            "photon_energy_bins": ebins,
            "effective_area": eff_area.astype(float) << u.cm**2,
        }
