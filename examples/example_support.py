import astropy.units as u
from sunkit_spex.extern import rhessi
import numpy as np
from yaff import fitting

def adapt_rhessi_data(rl: rhessi.RhessiLoader) -> fitting.DataPacket:
    '''Convert the sunkit-spex format into a fitting.DataPacket'''
    # One-D arrays for energy bins
    count_ebins = np.unique(rl['count_channel_bins'].flatten()) << u.keV
    photon_ebins = np.unique(rl['photon_channel_bins'].flatten()) << u.keV

    cts = rl._loaded_spec_data['counts'] << u.ct # (rl._spectrum['counts'] / rl._spectrum['livetime'])
    err = rl._loaded_spec_data['count_error'] << u.ct # (rl._spectrum['counts_err'] / rl._spectrum['livetime'])

    bkg_cts = (
        rl['extras']['background_rate'] *
        rl['count_channel_binning'] *
        rl['effective_exposure']
    )
    bkg_err = (
        rl['extras']['background_rate_error'] *
        rl['count_channel_binning'] *
        rl['effective_exposure']
    )

    effective_exposure = rl._loaded_spec_data['effective_exposure'] << u.s
    return fitting.DataPacket(
        counts=cts << u.ct,
        counts_error=err << u.ct,
        background_counts=bkg_cts << u.ct,
        background_counts_error=bkg_err << u.ct,
        effective_exposure=effective_exposure << u.s,
        count_energy_edges=count_ebins << u.keV,
        photon_energy_edges=photon_ebins << u.keV,
        response_matrix=rl['srm'].T << (u.cm**2 * u.ct / u.ph )
    )


def thermal_and_thick(arg_dict: dict[str, object]):
    # Imports need to be inside the model
    # function for pickling/multiprocessing
    from sunkit_spex.legacy import thermal
    from sunkit_spex.legacy import emission

    # The dict type annotation in the function
    # declaration is ambiguous; so, annotate the variables here
    ph_edges: np.ndarray = arg_dict['photon_energy_edges']
    params: dict[str, fitting.Parameter] = arg_dict['parameters']

    # Hack the thermal emission to work with
    # the RHESSI energy range (...)
    lowe, highe = thermal.CONTINUUM_GRID['energy range keV']
    ph_mids = ph_edges[:-1] + np.diff(ph_edges)/2
    kept = (ph_edges >= lowe) & (ph_edges <= highe)
    kept_mids = (ph_mids >= lowe) & (ph_mids <= highe)

    thermal_portion = thermal.thermal_emission(
        energy_edges=ph_edges[kept] << u.keV,
        temperature=params['temperature'].as_quantity(),
        emission_measure=params['emission_measure'].as_quantity()
    ).to_value(u.ph / u.s / u.keV / u.cm**2)

    # Zero-out the data which falls outside the
    # normal thermal energy grid
    thermal_fixed = np.zeros(kept.size - 1)
    thermal_fixed[kept_mids] = thermal_portion
    thermal_portion = thermal_fixed

    # We have to evaluate the thick target model at energy midpoints,
    # so compute them here
    ph_mids = ph_edges[:-1] + np.diff(ph_edges)/2

    # Single power law; very high energy break
    break_energy = 100 * ph_mids[-1]
    nonthermal_portion = emission.bremsstrahlung_thick_target(
        photon_energies=ph_mids,
        p=params['spectral_index'].value,
        q=0,
        eelow=params['cutoff_energy'].value,
        eebrk=break_energy,
        eehigh=break_energy,
    )

    # For this model, assume earth observer distance
    scaled_flux = params['electron_flux'].as_quantity().to_value(1e35 * u.electron / u.s)
    return thermal_portion + (1e35 * scaled_flux * nonthermal_portion)