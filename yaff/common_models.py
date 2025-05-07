"""Common models used when fitting X-ray data"""
from typing import TypeAlias
import astropy.units as u
from yaff import fitting
import numpy as np

# The type annotation for what's passed to different functions
ArgsT: TypeAlias = dict[str, np.ndarray | dict[str, fitting.Parameter]]


def thermal(args: dict[str, ArgsT]):
    '''Thermal bremsstrahlung emission, assuming coronal densities.'''
    from sunkit_spex.legacy import thermal

    params: dict[str, fitting.Parameter] = args['parameters']
    temperature = params['temperature']
    emission_measure = params['emission_measure']

    ph_edges: np.ndarray = args['photon_energy_edges']
    
    # Zero out anything outside the range of the current grid
    lowe, highe = thermal.CONTINUUM_GRID['energy range keV']
    ph_mids = ph_edges[:-1] + np.diff(ph_edges)/2
    kept = (ph_edges >= lowe) & (ph_edges <= highe)
    kept_mids = (ph_mids >= lowe) & (ph_mids <= highe)

    thermal_portion = thermal.thermal_emission(
        energy_edges=ph_edges[kept] << u.keV,
        temperature=temperature.as_quantity(),
        emission_measure=emission_measure.as_quantity()
    ).to_value(u.ph / u.s / u.keV / u.cm**2)

    # Perform the zeroing-out
    thermal_fixed = np.zeros(kept.size - 1)
    thermal_fixed[kept_mids] = thermal_portion
    thermal_portion = thermal_fixed
    return thermal_portion


def thick_target(args: dict[str, ArgsT]):
    '''A single power law thick target emission function.'''
    from sunkit_spex.legacy import emission

    ph_edges: np.ndarray = args['photon_energy_edges']
    # We have to evaluate the thick target model at energy midpoints,
    # so compute them here
    ph_mids = ph_edges[:-1] + np.diff(ph_edges)/2

    params: dict[str, fitting.Parameter] = args['parameters']
    spectral_index = params['spectral_index']
    electron_flux = params['electron_flux']
    cutoff_energy = params['cutoff_energy']

    # Single power law; very high energy break
    break_energy = 100 * ph_mids[-1]
    nonthermal_portion = emission.bremsstrahlung_thick_target(
        photon_energies=ph_mids,
        p=spectral_index.value,
        q=0,
        eelow=cutoff_energy.value,
        eebrk=break_energy,
        eehigh=break_energy,
    )

    # For this model, assume earth observer distance
    scaled_flux = electron_flux.as_quantity().to_value(
        1e35 * u.electron / u.s
    )
    return scaled_flux * nonthermal_portion


def broken_power_law(args: dict):
    '''a power law in photon space.
       param names for the dictionary:
       - norm_energy (usually fixed)
       - norm_flux
       - break_energy
       - lower_index (sometimes fixed)
       - upper_index
    '''
    from sunkit_spex.legacy import photon_power_law as ppl
    edges = args['photon_energy_edges'] << u.keV
    params = args['parameters']
    return ppl.compute_broken_power_law(
        edges,
        norm_energy=params['norm_energy'].as_quantity(),
        norm_flux=params['norm_flux'].as_quantity(),
        break_energy=params['break_energy'].as_quantity(),
        lower_index=params['lower_index'].as_quantity(),
        upper_index=params['upper_index'].as_quantity(),
    ).to_value(u.ph / u.keV / u.cm**2 / u.s)
