"""Spatial Consistency Tests"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from . import calculate as calc
from sklearn import linear_model

def hubbard_consistency(lat, lon, x, start_test, end_test, n_max, t_max, search_radius,
                        min_neigh, missing_perc, f, flag_val):
    r"""
    Hubbard spatial weighted regression analysis.

    Parameters
    ------------
    lat : array_like, shape(n,)
            Array of latitudes in decimal degrees
    lon : array_like, shape(n,)
            Array of longitudes in decimal degrees
    x   : pd.DataFrame, shape(t,n)
            Pandas dataframe of observations, where t is time and n is the number of stations
    start_test: string
            Datetime string indicating when to start testing
    end_test: string
            Datetime string indicating when to end testing
    n_max   : int
            Maximum number of best fit stations to use for the estimate
    t_max   : int
            Number of time steps to be considered in the regression analysis (even number)
    search_radius: float
            Radius for Hubbard analysis in decimal degrees
    min_neigh: int
            Minimum number of neighbors to find the estimate
    missing_perc: int
            Maximum percentage of missing data to perform regression
    f        : int
            Factor multiplying standard deviation for calculating the acceptable range for
            valid observations.
    flag_val : int
            Integer representing flag values to be associated to
            erroneous values.

    Returns
    -----------
    df_x_est: pd.DataFrame, shape(t,n)
            Estimated observations,
            filled with np.nan values where estimate is not possible
    df_std_est: pd.DataFrame, shape(t,n)
            Standard deviation from the estimated observations,
            filled with np.nan values where estimate is not possible
    flag    : pd.DataFrame, shape(t,n)
            Dataframe with flags identifying values which fail the test.
    res     : pd.DataFrame, shape(t,n)
            Dataframe with quantitative residuals from prescribed limits:
            positive values indicates wrong values.
    Notes
    -----
    The spatial weighted regression test is based on the algorithm proposed by [Hubbard2005]_.
    Firstly for each reference station :math:`(0)` the neighbour stations :math:`(n)` inside a certain
    `search_radius` are founded. This search radius should be set close to the average spacing
    of the observations and large enough to have at least one neighbour for each station.

    Once the neighbours have been established, if the number of missing values in each series
    is lower than missing_perc and n :math:`\ge` `n_max` , a linear regression  is computed
    between their values :math:`x(t,n)` over all the selected time steps `t_max` and
    the reference station values :math:`x(t,0)`, in order to find a first estimate
    :math:`x^*_n(0)` of :math:`x(0)` that should be consistent with :math:`x(n)` at each time step.

    Then the root mean square error between the reference values :math:`x(0)` and
    the estimated values :math:`x^*_n(0)` from the regression line with the neighbor station :math:`n`
    (correspondent to the sample standard deviation of the residuals :math:`\sigma^*_n` )
    is evaluated in order to find a measure of the stations correlation:

    .. math::

        \sigma^*_n(0)=\sqrt{\frac{1}{t_{max}}\sum_{t=1}^{t_{max}}
        \big[\underbrace{x(t,0)-x^*_n(t,0)}_{\text{Residuals}}\big]^2}

    This error characterizing each neighbour station is used as weight in the final estimate
    of the reference value :math:`x^*(t,0)` and reference standard deviation :math:`\sigma^*(0)`
    from the surrounding stations at each instant t:

    .. math::

       x^*(t,0)=\frac{\sum_{n=1}^{n_{max}} (x^*_n)^2/(\sigma^*_n)^2}{\sum_{n=1}^{n_{max}} 1/\sigma_n^2} \\
       \sigma^{*2}(0)= \frac{n_{max}}{\sum_{n=1}^{n_{max}} 1/(\sigma^*_n)^2}

    Finally at each time step a tolerance interval is established by considering
    a constant factor `f` and the spatial consistency is verified by ensuring that:

    .. math::

        x^*(t,0)-f\sigma^*(0) < x(t,0) < x^*(t,0)+f\sigma^*(0)

    This procedure is repeated for each station and each time step. The final estimates
    :math:`x^*(t,n)` and reference standard deviation :math:`\sigma^*(n)` are given as results on the
    output dataframes `df_x_est` and `df_std_est`, respectively.

    References
    ----------
    .. [Hubbard2005] Hubbard, K. G., et al. "Performance of quality assurance procedures for an applied climate information system. "Journal of Atmospheric and Oceanic Technology 22.1 (2005): 105-112.
    """

    print('Minimum number of Neighbors: {}'.format(min_neigh))
    print('Search radius in decimal degrees: {}'.format(search_radius))

    t, n = x.shape

    time_ind = pd.Series(np.arange(0, t), x.index)
    x_est = np.full(((time_ind[end_test] - time_ind[start_test]) + 1, n), np.nan)
    std_est = np.full(((time_ind[end_test] - time_ind[start_test]) + 1, n), np.nan)

    if t < t_max:
        print('Warning: Specified interval larger than time series length')
        print('Statistics could be not accurate')

    # -------------------------------------------------------------------#
    # Start cycle for each station
    # -------------------------------------------------------------------#
    for stat in tqdm(range(0, n)):
        # ----- Find neighbors inside a search radius -------------------#
        ind_neigh = calc.find_neighbors(np.vstack((np.delete(lat, stat),
                                                   np.delete(lon, stat))).T,
                                        np.vstack((lat[stat], lon[stat])).T, search_radius)
        # ----- Start estimate only if there is a minimum number of neighbors -----#
        if len(ind_neigh) < min_neigh:
            print('\n Number of neighbors less than {} for station {}'.format(min_neigh, x.columns[stat]))
        else:
            x_neigh = np.delete(x.values, stat, axis=1)[:, ind_neigh]

            # ---------------------------------------------------------------#
            # Start cycle for selected time steps---------------------------#
            # ---------------------------------------------------------------#
            t1 = 0
            for time in range(time_ind[start_test], time_ind[end_test] + 1):
                # --------------------------------------------------------
                # Extracting time interval to consider for regression
                # --------------------------------------------------------
                if time < t_max:
                    x_sel = x_neigh[:time + t_max, :]
                    x_stat = x.iloc[:time + t_max, stat].values
                elif time > (t - t_max):
                    x_sel = x_neigh[time - t_max:, :]
                    x_stat = x.iloc[time - t_max:, stat].values
                else:
                    x_sel = x_neigh[time - int(t_max / 2):time + int(t_max / 2), :]
                    x_stat = x.iloc[time - int(t_max / 2):time + int(t_max / 2), stat].values

                # --------------------------------------------------------
                # Ensuring a certain percentage of valid data and
                # a minimum number of valid neighbors
                # --------------------------------------------------------

                miss = (np.isnan(x_stat).sum() / (len(x_stat))) * 100
                miss_neigh = (np.isnan(x_sel).sum(axis=0) / (len(x_sel))) * 100
                ind_valid_neigh = np.where(miss_neigh < missing_perc)[0]

                if (miss <= missing_perc) & (len(ind_valid_neigh) >= min_neigh):
                    # ----------------------------------------------------
                    # Starting estimate ---------------------------------
                    # ----------------------------------------------------
                    y_est_t = np.full(ind_valid_neigh.shape, np.nan)  # Instantaneous estimate at the chosen time step
                    rmse = np.full(ind_valid_neigh.shape, np.nan)  # rmse evaluated considering the chosen interval
                    j = 0
                    # ---------------------------------------------------
                    # Iterate over each neighbor
                    # ---------------------------------------------------
                    for i in ind_valid_neigh:
                        mask_nan = ((np.isnan(x_sel[:, i])) | (np.isnan(x_stat)))
                        regr = linear_model.LinearRegression()
                        regr.fit(x_sel[:, i][~mask_nan].reshape(-1, 1), x_stat[~mask_nan])
                        # Local estimate of the selected station
                        # from the selected neighbors
                        if ~np.isnan(x_neigh[time, i]):
                            y_est_t[j] = regr.predict(x_neigh[time, i].reshape(-1, 1))
                        y_est_tot = regr.predict(x_sel[:, i][~mask_nan].reshape(-1, 1))
                        rmse[j] = np.maximum(0.0001, np.sqrt(((y_est_tot - x_stat[~mask_nan]) ** 2).mean()))

                        j += 1

                    # ----------------------------------------------------
                    # Select only N best fit neighbors
                    # ----------------------------------------------------
                    ind_sel_neigh = np.argsort(rmse)[:n_max]

                    # ----------------------------------------------------
                    # Remove nan estimates
                    # ----------------------------------------------------
                    mask_nan = np.isnan(y_est_t[ind_sel_neigh])
                    ind_sel_neigh = ind_sel_neigh[~mask_nan]
                    # ----------------------------------------------------
                    # Calculate final estimate only if enough
                    # neighbors estimate are present
                    # ----------------------------------------------------
                    if len(ind_sel_neigh) >= min_neigh:
                        # Divide into positive and negative contributions
                        if np.all(np.sign(y_est_t[ind_sel_neigh]) == -1):
                            # Estimate all neg
                            x_est[t1, stat] = -1 * np.sqrt(
                                np.nansum(y_est_t[ind_sel_neigh] ** 2 / rmse[ind_sel_neigh] ** 2) / np.nansum(
                                    1 / rmse[ind_sel_neigh] ** 2))
                        elif np.all(np.sign(y_est_t[ind_sel_neigh]) >= 0):
                            # Estimate all pos
                            x_est[t1, stat] = np.sqrt(
                                np.nansum(y_est_t[ind_sel_neigh] ** 2 / rmse[ind_sel_neigh] ** 2) / np.nansum(
                                    1 / rmse[ind_sel_neigh] ** 2))
                        else:
                            # Estimate mixed type
                            mask_pos = y_est_t[ind_sel_neigh] >= 0
                            # X_est_pos=np.sqrt(np.nansum(y_est_t[ind_sel_neigh][mask_pos]**2/rmse[ind_sel_neigh][mask_pos]**2)/np.nansum(1/rmse[ind_sel_neigh][mask_pos]**2))
                            # X_est_neg=-1*np.sqrt(np.nansum(y_est_t[ind_sel_neigh][~mask_pos]**2/rmse[ind_sel_neigh][~mask_pos]**2)/np.nansum(1/rmse[ind_sel_neigh][~mask_pos]**2))
                            x_est_pos = np.sqrt(np.nansum(
                                y_est_t[ind_sel_neigh][mask_pos] ** 2 / rmse[ind_sel_neigh][mask_pos] ** 2) / np.nansum(
                                1 / rmse[ind_sel_neigh] ** 2))
                            x_est_neg = -1 * np.sqrt(np.nansum(
                                y_est_t[ind_sel_neigh][~mask_pos] ** 2 / rmse[ind_sel_neigh][
                                    ~mask_pos] ** 2) / np.nansum(1 / rmse[ind_sel_neigh] ** 2))

                            x_est[t1, stat] = x_est_pos + x_est_neg

                        # Standard error
                        std_est[t1, stat] = np.sqrt(1 / np.nanmean(rmse[ind_sel_neigh] ** 2))

                t1 += 1
    # Saving pandas dataframe
    df_x_est = pd.DataFrame(x_est, columns=x.columns,
                            index=time_ind[start_test:end_test].index)
    df_std_est = pd.DataFrame(std_est, columns=x.columns,
                              index=time_ind[start_test:end_test].index)
    flag = pd.DataFrame(0, index=time_ind[start_test:end_test].index, columns=x.columns, )
    res = pd.DataFrame(0, index=time_ind[start_test:end_test].index, columns=x.columns, )

    min_val = df_x_est - f * df_std_est
    max_val = df_x_est + f * df_std_est

    mask_spatial = ((x[start_test:end_test] > max_val) | (x[start_test:end_test] < min_val))
    flag[mask_spatial] = flag_val

    res_min = min_val - x[start_test:end_test]
    res_max = x[start_test:end_test] - max_val

    mask_res = (res_min >= res_max)
    res[mask_res] = res_min[mask_res]
    res[~mask_res] = res_max[~mask_res]

    return df_x_est, df_std_est, flag, res

