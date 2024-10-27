import numpy as np
from numpy.typing import ArrayLike


'''
Perform a flux- (i.e. area-) conserving rebinning of a histogram,
given its edges and a set of new edges.

The __name__ ... section has an example using a power law.
'''


def flux_conserving_rebin(
    old_edges: ArrayLike,
    old_values: ArrayLike,
    new_edges: ArrayLike,
) -> np.ndarray:
    '''
    Rebin a histogram by performing a flux-conserving rebinning.
    The total area of the histogram is conserved.
    Adjacent bins are proportionally interpolated for new edges that do not line up.
    Don't make the new edges too finely spaced;
    don't make a new bin fall inside of an old one completely.
    '''
    old_edges = np.array(np.sort(old_edges))
    new_edges = np.array(np.sort(new_edges))
    nd = np.diff(new_edges)
    od = np.diff(old_edges)

    if (new_edges[0] < old_edges[0]) or (new_edges[-1] > old_edges[-1]):
        raise ValueError('New edges cannot fall outside range of old edges.')

    try:
        if np.all(nd == od):
            return old_values
    except ValueError:
        pass

    orig_flux = od * old_values
    ret = np.zeros(new_edges.size - 1)
    for i in range(ret.size):
        ret[i] = interpolate_new_bin(
            original_area=orig_flux,
            old_edges=old_edges,
            new_left=new_edges[i],
            new_right=new_edges[i+1]
        )

    return ret


def proportional_interp_single_bin(
    left_edge: float,
    right_edge: float,
    interp: float
) -> tuple[float, float]:
    '''
    say what portion of a histogram bin belongs on the left and right
    of an edge to interpolate.
    '''
    denom = right_edge - left_edge
    right_portion = (right_edge - interp) / denom
    left_portion = (interp - left_edge) / denom
    return left_portion, right_portion


def bounding_interpolate_indices(
    old_edges: np.ndarray,
    left: float,
    right: float
) -> tuple[int, int]:
    '''
    find the indices of the old edges that bound the new left
    and right edges.
    '''
    indices = np.arange(old_edges.size)
    new_left = indices[old_edges <= left][-1]
    new_right = indices[old_edges >= right][0]
    return (new_left, new_right)


def interpolate_new_bin(
    original_area: np.array,
    old_edges: np.array,
    new_left: float,
    new_right: float
) -> float:
    '''
    interpolate the new bin value given old edges, new edges,
    and the old flux (aka area).
    '''
    oa = original_area
    oe = old_edges

    old_start_idx, old_end_idx = bounding_interpolate_indices(
        oe, new_left, new_right
    )

    # portion of edge bins that get grouped with the new bin
    _, left_partial_prop = proportional_interp_single_bin(
        left_edge=oe[old_start_idx],
        right_edge=oe[old_start_idx+1],
        interp=new_left
    )
    left_partial_area = left_partial_prop * oa[old_start_idx]

    right_partial_prop, _ = proportional_interp_single_bin(
        left_edge=oe[old_end_idx-1],
        right_edge=oe[old_end_idx],
        interp=new_right
    )
    right_partial_area = right_partial_prop * oa[old_end_idx-1]

    partial_slice = slice(old_start_idx + 1, old_end_idx - 1)
    have_bad_slice = (partial_slice.start > partial_slice.stop)
    if have_bad_slice:
        raise ValueError('Your new bins are too fine. Use coarser bins.')

    between_area = oa[partial_slice].sum()

    delta = new_right - new_left
    new_bin_value = (left_partial_area + between_area + right_partial_area) / delta
    return new_bin_value


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def power_law(x, x0, idx):
        return (x / x0) ** idx


    def compute_binned_flux(edges, values):
        return np.sum(values * np.diff(edges))

    bin_edges = np.linspace(1, 100, num=1000)
    new_edges = np.logspace(0, 2, num=40)
    mids = bin_edges[:-1] + np.diff(bin_edges)/2
    x0 = 10
    idx = -3
    values = power_law(mids, x0, idx)

    new_values = flux_conserving_rebin(
        old_edges=bin_edges,
        old_values=values,
        new_edges=new_edges
    )

    original_flux = compute_binned_flux(bin_edges, values)

    new_flux = compute_binned_flux(new_edges, new_values)

    print(f'original flux:     {original_flux:.4f}')
    print(f'rebinned flux:     {new_flux:.4f}')
    print(f'amount off:        {np.abs(original_flux - new_flux):.2e}')
    print(f'machine precision: {np.finfo(float).eps:.2e}')
    print(';)')

    fig, ax = plt.subplots(figsize=(8, 6), layout='constrained')
    ax.stairs(values, bin_edges, label='original')
    ax.stairs(new_values, new_edges, label='rebinned')
    ax.set(
        xscale='log',
        yscale='log',
        xlabel='$x$',
        ylabel='power law',
        title='Rebin linear-spaced bins to log-spaced bins'
    )
    ax.legend()
    plt.show()
