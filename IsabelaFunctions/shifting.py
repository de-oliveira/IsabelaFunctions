import numpy as np


def B_sum(BM, BO):
    """ Equation 3.2 in Master Dissertation.

    Parameters:
        BM: array
            Model magnetic field.

        BO: array
            Observed magnetic field (rolled or not).

    Returns:
        total_sum: float
            Sum of the absolute difference from the subtraction of the two magnetic fields, normalized by the number of data points.
    """
    res = abs(BM - BO)
    count = np.count_nonzero(~np.isnan(res))
    total_sum = np.nansum(res)/count
    return total_sum


def shifting_technique(BM, BO, N=10):
    """ Equation 3.2 in Master Dissertation, with shifting applied

   Parameters:
        BM: array
            Model magnetic field.

        BO: array
            Observed magnetic field.

        N: integer
            Total number of positive longitudinal shifts.

    Returns:
        total_sum: float
            B_sum for a range of longitudinal shifts
    """
    total_sum = np.array([B_sum(BM, np.roll(BO, i, axis=1)) for i in np.arange(-N, N+1)])
    return total_sum
