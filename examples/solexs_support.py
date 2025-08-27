from astropy.io import fits
import numpy as np
import astropy.units as u

def read_rmf(rmf_fn: str) -> dict[str, u.Quantity]:
    '''
    Read in a SoLEXS RMF file.
    The RMF file describes only how photon energies get distributed
    probabilistically into count energy bins (e.g. the energy resolution component).
    It does not contain effective area information.
    '''
    with fits.open(rmf_fn) as rd:
        count_ebins = np.concatenate((
            rd[1].data['E_MIN'],
            [rd[1].data['E_MAX'][-1]]
        )) << u.keV
        matrix = rd[2].data['MATRIX']
        return {
            'count_energy_bins': count_ebins << u.keV,
            'redistribution_matrix': matrix.T << (u.ct / u.ph)
        }


def read_arf(arf_fn: str) -> dict[str, u.Quantity]:
    with fits.open(arf_fn) as f:
        dat = f[1].data
        ebins = np.concatenate(
            (dat['energ_lo'], [dat['energ_hi'][-1]]),
            dtype=float
        ) << u.keV
        eff_area = dat['specresp']
        return {
            'photon_energy_bins': ebins,
            'effective_area': eff_area.astype(float) << u.cm**2
        }