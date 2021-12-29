""" Temporal consistency tests"""

import pandas as pd
import numpy as np

def step_all(x, step_val_susp, step_val_wrong, flag_val_susp, flag_val_wrong):
    r"""
        Check the difference between consecutive observations.

        Parameters
        ----------
        x : pd.DataFrame
                Dataframe to be tested (time, stations)
        step_val_susp : float
                step limit for suspect values.
        step_val_wrong : float
                step limit for wrong values.
        flag_val_susp: int
                integer representing flag values to be associated to
                suspect values.
        flag_val_wrong: int
                integer representing flag values to be associated to
                wrong values.

        Returns
        -------
        flag : pd.DataFrame
                Dataframe with flags identifying values which fail the test.
        res : pd.DataFrame
                Dataframe with quantitative residuals from prescribed limits:
                positive values indicates suspect or wrong values.

        Warning
        -------
        This function must be used with dataframes with homogeneous temporal resolution
        on the temporal index. Missing dates have to be filled with np.nan values.

        """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns, )
    res = pd.DataFrame(0, index=x.index, columns=x.columns, )

    prev = x.shift(-1)
    post = x.shift(+1)

    diff_prev = abs(x - prev)
    diff_post = abs(post - x)

    mask_step_wrong = ((diff_prev > step_val_wrong) | (diff_post > step_val_wrong))
    mask_step_susp = (((diff_prev > step_val_susp) | (diff_post > step_val_susp)) & (~mask_step_wrong))

    res_prev = diff_prev - step_val_susp
    res_post = diff_post - step_val_susp

    mask_res = (res_prev >= res_post)
    res[mask_res] = res_prev[mask_res]
    res[~mask_res] = res_post[~mask_res]

    flag[mask_step_susp] += flag_val_susp
    flag[mask_step_wrong] += flag_val_wrong

    return flag, res


def step_seas(x, step_vals_susp, step_vals_wrong, flag_val_susp, flag_val_wrong):
    r"""
        Check the difference between consecutive observations by considering different
        limits for each season.

        Parameters
        ----------
        x : pd.DataFrame
                Dataframe to be tested (time, stations)
        step_vals_susp : array_like, shape(4,)
                Array of step limits for suspect values for each season in this order:
                DJF (December, January, February),MAM (March, April, May),
                JJA (June, July, August), SON (September, October, November).
        step_vals_wrong : array_like, shape(4,)
                Array of step limits for wrong values for each season in this order:
                DJF (December, January, February),MAM (March, April, May),
                JJA (June, July, August), SON (September, October, November).
        flag_val_susp: int
                integer representing flag values to be associated to
                suspect values.
        flag_val_wrong: int
                integer representing flag values to be associated to
                wrong values.

        Returns
        -------
        flag : pd.DataFrame
                Dataframe with flags identifying values which fail the test.
        res : pd.DataFrame
                Dataframe with quantitative residuals from prescribed limits:
                positive values indicates suspect or wrong values.

        Warning
        -------
        This function must be used with dataframes with homogeneous temporal resolution
        on the temporal index. Missing dates have to be filled with np.nan values.

        """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns, )
    res = pd.DataFrame(0, index=x.index, columns=x.columns, )

    mask_djf = ((x.index.month == 12) | (x.index.month == 1) | (x.index.month == 2))
    mask_mam = ((x.index.month == 3) | (x.index.month == 4) | (x.index.month == 5))
    mask_jja = ((x.index.month == 6) | (x.index.month == 7) | (x.index.month == 8))
    mask_son = ((x.index.month == 9) | (x.index.month == 10) | (x.index.month == 11))

    mask_seas = [mask_djf, mask_mam, mask_jja, mask_son]

    for i in range(len(step_vals_susp)):
        x_seas = x.copy()
        x_seas[~mask_seas[i]] = np.nan
        flag_seas, res_seas = step_all(x_seas, step_vals_susp[i], step_vals_wrong[i],
                                       flag_val_susp, flag_val_wrong)
        flag[mask_seas[i]] = flag_seas
        res[mask_seas[i]] = res_seas

    return flag, res


def persistence_noc(x, n, flag_val):
    r"""
        Not Observed Change (NOC) Persistence test to check minimum variability
        of a certain variable with respect to the previous n time steps.

        Parameters
        ----------
        x : pd.DataFrame
                Dataframe to be tested (time, stations)
        n : int
                Number of previous time steps to be considered in the test.
                The value must be bigger than 1.
        flag_val: int
                integer representing flag values to be associated to suspect values.

        Returns
        -------
        flag : pd.DataFrame
                Dataframe with flags identifying values which fail the test.
        res : pd.DataFrame
                Dataframe with quantitative residuals representing
                the maximum absolute difference between the selected observation and the
                preceding ones. As this value tends to zero, the selected observation
                indicates temporal persistence.

        Warning
        -------
        This function must be used with dataframes with homogeneous temporal resolution
        on the temporal index. Missing dates have to be filled with np.nan values.

        """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns, )

    prev1 = x.shift(+1)
    diff1 = abs(x - prev1)
    res = diff1.copy()
    mask_pers = (diff1 == 0)

    for i in range(2, n+1):

        prev_i = x.shift(+i)
        diff_i = abs(x - prev_i)
        mask_pers = (mask_pers & (diff_i == 0))
        res = np.maximum(res, diff_i)

    flag[mask_pers] += flag_val

    return flag, res


def persistence_var(x, window, perc_min, method, var_val_min, flag_val):
    r"""
        Minimum Variability Persistence test to check minimum allowed variability
        of a certain variable with respect to a certain time window.

        Parameters
        ----------
        x : pd.DataFrame
                Dataframe to be tested (time, stations)
        window : int
                Number of hours over which evaluate temporal variability
        perc_min : int
                Minimum percentage of valid observations (not missing)
                in order to calculate temporal variability
        method : {'STD', 'MAX_MIN', 'IQR'}
                Method for calculating the temporal variability within the desired
                time window: standard deviation (STD), difference between absolute maximum
                and minimum values (MAX_MIN), interquartile range (IQR).
        var_val_min : float
                Minimum allowed variability for the selected variable within the
                defined time window
        flag_val: int
                integer representing flag values to be associated to suspect values.

        Returns
        -------
        flag : pd.DataFrame
                Dataframe with flags identifying values which fail the test.
        res : pd.DataFrame
                Dataframe with quantitative residuals representing
                the difference between the actual variability and the minimum allowed
                variability. Positive values indicates suspect observations.

        Raises
        ------
        Exception
            If the selected method is neither 'STD', 'MAX_MIN' nor 'IQR'.
        """
    flag = pd.DataFrame(0, index=x.index, columns=x.columns, )
    var_count = x.resample('{}H'.format(window)).count()
    var_perc = (var_count / window) * 100
    mask_perc = (var_perc <= perc_min)

    if method == 'STD':
        std = x.resample('{}H'.format(window)).std()

        std_nan = std.copy()
        std_nan[mask_perc] = np.nan

        std_res = std_nan.resample('H').pad()

        mask_nan = np.isnan(x)
        mask_pers_std = ((std_res < var_val_min) & (~mask_nan))
        flag[mask_pers_std] += flag_val

        res = var_val_min - std_res

    elif method == 'MAX_MIN':
        r = x.resample('{}H'.format(window)).max() - x.resample('{}H'.format(window)).min()

        r_nan = r.copy()
        r_nan[mask_perc] = np.nan

        r_res = r_nan.resample('H').pad()

        mask_nan = np.isnan(x)
        mask_pers_maxmin = ((r_res < var_val_min) & (~mask_nan))
        flag[mask_pers_maxmin] += flag_val

        res = var_val_min - r_res

    elif method == 'IQR':
        r = x.resample('{}H'.format(window)).quantile(0.75) - x.resample('{}H'.format(window)).quantile(0.25)

        r_nan = r.copy()
        r_nan[mask_perc] = np.nan

        r_res = r_nan.resample('H').pad()

        mask_nan = np.isnan(x)
        mask_pers_iqr = ((r_res < var_val_min) & (~mask_nan))
        flag[mask_pers_iqr] += flag_val

        res = var_val_min - r_res

    else:
        raise Exception('Selected method not found. Available methods are:'
                        ' STD, MAX_MIN and IQR.')

    return flag, res


def isolated(x, n, flag_val):
    r"""
        Check if an observation of rainfall (or other discrete variables) is isolated with
        respect to the adjacent measures (in time).

        Parameters
        ----------
        x : pd.DataFrame
                Dataframe to be tested (time, stations)
        n : int
                Number of previous and following time steps to be considered in the test.
                The value must be bigger than 1.
        flag_val: int
                integer representing flag values to be associated to suspect values.

        Returns
        -------
        flag : pd.DataFrame
                Dataframe with flags identifying values which fail the test.
        res : pd.DataFrame
                Dataframe with quantitative residuals representing
                the difference between the selected observation and the
                sum of the adjacent ones. As this value increases, the selected observation
                is more isolated with respect to neighbors.

        Warning
        -------
        This function must be used with dataframes with homogeneous temporal resolution
        on the temporal index. Missing dates have to be filled with np.nan values.

        """

    flag = pd.DataFrame(0, index=x.index, columns=x.columns, )

    prev1 = x.shift(+1)
    post1 = x.shift(-1)
    sum = prev1+post1
    mask_isol = ((x != 0) & (sum == 0))

    for i in range(2, n+1):

        prev_i = x.shift(+i)
        post_i = x.shift(-i)
        sum_i = prev_i + post_i
        mask_isol = (mask_isol & (sum_i == 0))
        sum = sum + sum_i

    res = x - sum
    flag[mask_isol] += flag_val

    return flag, res