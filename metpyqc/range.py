"""Range tests"""
import numpy as np
import pandas as pd

def range_all(x, min_val, max_val, flag_val):
    r"""
    Check if observed values are outside the instrumental limits.

    Parameters
    ----------
    x : pd.DataFrame
            Dataframe to be tested (time, stations)
    min_val : float
            lower limit of instrumental range.
    max_val : float
            upper limit of instrumental range.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying values which fail the test.
    res : pd.DataFrame
            Dataframe with quantitative residuals from prescribed limits:
            positive values indicates wrong values.

    """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns,)
    res = pd.DataFrame(0, index=x.index, columns=x.columns, )

    mask = ((x > max_val) | (x < min_val))

    flag[mask] = flag_val
    res_min = min_val-x
    res_max = x-max_val

    mask_res = (res_min >= res_max)
    res[mask_res] = res_min[mask_res]
    res[~mask_res] = res_max[~mask_res]

    return flag, res

def range_seas(x, min_vals, max_vals, flag_val):
    r"""
    Check if observed values are outside the seasonal climatological limits.

    Parameters
    ----------
    x : pd.DataFrame
            Dataframe to be tested (time, stations)
    min_vals : array_like, shape(4,)
            Array of lower limits for each season in this order: DJF (December, January, February),
            MAM (March, April, May), JJA (June, July, August), SON (September, October, November).
    max_vals : array_like, shape(4,)
            Array of upper limits for each season in this order: DJF (December, January, February),
            MAM (March, April, May), JJA (June, July, August), SON (September, October, November).
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying values which fail the test.
    res : pd.DataFrame
            Dataframe with quantitative residuals from prescribed limits:
            positive values indicates wrong values.

    """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns,)
    res = pd.DataFrame(0, index=x.index, columns=x.columns, )

    mask_djf = ((x.index.month == 12) | (x.index.month == 1) | (x.index.month == 2))
    mask_mam = ((x.index.month == 3) | (x.index.month == 4) | (x.index.month == 5))
    mask_jja = ((x.index.month == 6) | (x.index.month == 7) | (x.index.month == 8))
    mask_son = ((x.index.month == 9) | (x.index.month == 10) | (x.index.month == 11))

    mask_seas = [mask_djf, mask_mam, mask_jja, mask_son]

    for i in range(len(max_vals)):
        x_seas = x.copy()
        x_seas[~mask_seas[i]] = np.nan
        flag_seas, res_seas = range_all(x_seas, min_vals[i], max_vals[i], flag_val)
        flag[mask_seas[i]] = flag_seas
        res[mask_seas[i]] = res_seas

    return flag, res
