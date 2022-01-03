""" Useful routines for quality control tests"""
import numpy as np
from scipy.spatial import cKDTree


def dewpoint(temperature, relative_humidity):
    r"""
    Calculate dewpoint temperature from temperature and relative humidity.

    Parameters
    ----------
    temperature : pd.DataFrame
            Temperature in degree Celsius
    relative_humidity : pd.DataFrame
            Relative humidity in percentage

    Returns
    -------
    dew : pd.DataFrame
            Dewpoint temperature in degree Celsius

    Notes
    -----
    The formula used to calculate dewpoint temperature is obtained by
    inverting the Bolton's formula [Emanuel1994]_:

    .. math:: T_{d} \simeq \frac{243.5}{(\frac{17.67}{ln (e/6.112)}) -1}.

    where :math:`e = RH e^{*}`, with e being the actual vapor pressure, RH the relative humidity
    and :math:`e^{*}` the saturation vapor pressure as obtained from the [Bolton1980]_ formula:

    .. math:: e^{x} = 6.112 exp (\frac{17.67 T}{T +243.5})

    References
    ----------
    .. [Emanuel1994] Emanuel, Kerry A. Atmospheric convection. Oxford University Press on Demand, 1994.
    .. [Bolton1980] Bolton, David. The computation of equivalent potential temperature.Monthly weather review, 1980, 108.7: 1046-1053.

    """
    sat_vap_press = 6.112 * np.exp((17.67 * temperature) / (temperature + 243.5))
    vap_press = (relative_humidity/100) * sat_vap_press
    dew = (243.5 * np.log(vap_press / 6.112)) / (17.67 - np.log(vap_press / 6.112))
    return dew


def find_neighbors(points, xi, r):
    r"""
    Find neighbors point inside a search radius

    Parameters
    ----------
    points : array_like, shape(N,2)
            (lat,lon) of surrounding points in decimal degrees. Shape (N,2)
    xi     : array_like, shape(1,2)
            (lat,lon) of center of ball in decimal degrees. Shape (1,2)
    r      : float
            Search radius in decimal degrees

    Returns
    -------
    indices: array_like,
            Indices of neighbors points inside the search radius
    """
    obs_tree = cKDTree(points)
    indices = obs_tree.query_ball_point(xi.squeeze(), r=r)

    return indices