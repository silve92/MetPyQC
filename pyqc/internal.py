"""Internal Consistency Tests"""
import numpy as np
import pandas as pd

from . import calculate as calc

def maxmin(x, x_max, x_min, flag_val):
    r"""
    Check if averaged observed values lies between their maximum and minimum
    corresponding observations.

    Parameters
    ----------
    x, x_max, x_min : pd.DataFrame
            Dataframes to be tested (time, stations):
            average, maximum and minimum values respectively
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

    mask_nan = (~np.isnan(x) & ~np.isnan(x_max) & ~np.isnan(x_min))
    mask = ((x > x_max) | (x < x_min))

    flag[(mask_nan & mask)] = flag_val
    res_min = x_min-x
    res_max = x-x_max

    mask_res = (res_min >= res_max)
    res[mask_res] = res_min[mask_res]
    res[~mask_res] = res_max[~mask_res]

    return flag, res


def dewpoint_test(temp, rh, flag_val):
    r"""
    Check if derived dewpoint temperature is smaller or equal than temperature.

    Parameters
    ----------
    temp : pd.DataFrame
            Temperature dataframe to be tested (time, stations_temp) in degree Celsius
    rh   : pd.DataFrame
            Relative humidity dataframe to be tested (time, stations_rh) in percentage
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag_temp : pd.DataFrame
            Dataframe with flags identifying values which temperature value fail the test.
    flag_rh : pd.DataFrame
            Dataframe with flags identifying values which relative humidity
             value fail the test.
    res : pd.DataFrame
            Dataframe with quantitative residuals from prescribed limits:
            positive values indicates wrong values (use for temperature dataframe).
    temp_dew : pd.DataFrame
            Dew-point temperature in degree Celsius

    Warning
    -------
    This test is applied only to stations measuring both temperature and
    relative humidity.
    """
    stat_temp_rh = temp.columns.intersection(rh.columns)

    flag_temp = pd.DataFrame(0, index=temp.index, columns=temp.columns)
    flag_rh = pd.DataFrame(0, index=rh.index, columns=rh.columns)

    temp_dew = pd.DataFrame(np.round(calc.dewpoint(temp.loc[:, stat_temp_rh], rh.loc[:, stat_temp_rh]), 1),
                            columns=stat_temp_rh, index=rh.index)

    mask_dew = (temp_dew.loc[:, stat_temp_rh] >= temp.loc[:, stat_temp_rh])
    flag_temp[mask_dew] = flag_val
    flag_rh[mask_dew] = flag_val
    res = temp_dew - temp

    return flag_temp, flag_rh, res, temp_dew


def wspeed_wdir(ws, wd, flag_val):
    r"""
    Check if wind speed and direction are consistent: if wind speed is null,
    then also wind direction should be null and viceversa. Residuals cannot be evaluated
    from this logical condition.

    Parameters
    ----------
    ws : pd.DataFrame
            Wind speed dataframe to be tested (time, stations) in m/s
    wd   : pd.DataFrame
            Wind direction dataframe to be tested (time, stations) in degree north.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying values which fail the test.
    """
    flag = pd.DataFrame(0, index=ws.index, columns=ws.columns)

    mask_wswd0 = ((wd == 0) & (ws != 0) & (~np.isnan(ws)))
    mask_wswd1 = ((wd != 0) & (ws == 0) & (~np.isnan(wd)))
    flag[mask_wswd0 | mask_wswd1] = flag_val

    return flag

def heated_raingauge(prec, temp, ind_heater, flag_val):
    r"""
    Check when precipitation occurs with freezing temperatures (below 0 degree Celsius).
    Flag values when the raingauge is not heated.

    Parameters
    ----------
    prec : pd.DataFrame
            Precipitation dataframe to be tested (time, stations_prec) in mm
    temp : pd.DataFrame
            Temperature dataframe to be tested (time, stations_temp) in degree Celsius.
    ind_heater : list
            List of indexes indicating which raingauge is heated.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying precipitation values which fail the test.

    Warning
    -------
    This test is applied only to stations measuring both precipitation and
    temperature.
    """
    stat_temp_prec = prec.columns.intersection(temp.columns)

    flag = pd.DataFrame(0, index=prec.index, columns=prec.columns)

    mask_prectg = ((prec.loc[:, stat_temp_prec] > 0) & (temp.loc[:, stat_temp_prec] < 0))
    flag[mask_prectg] = flag_val
    # reset flag for raingauge with heater
    flag.iloc[:, ind_heater] = 0

    return flag

def snow_grass(snd, snd_toll, start_grass, end_grass, flag_val):
    r"""
    Check when snow depth exceed tolerance values during grass growth period.

    Parameters
    ----------
    snd : pd.DataFrame
            Snow depth dataframe to be tested (time, stations) in cm
    snd_toll : float
            Snow depth instrumental tolerance in cm
    start_grass : int
            Month number identifying the starting period of grass growth.
    end_grass : int
            Month number identifying the ending period of grass growth.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying precipitation values which fail the test.
    """
    flag = pd.DataFrame(0, index=snd.index, columns=snd.columns)
    ind_wrong = np.where((snd.index.month >= start_grass) & (snd.index.month <= end_grass))[0]

    mask_snd = (snd.iloc[ind_wrong, :] > snd_toll)
    flag[mask_snd] = flag_val

    return flag

def humidity_prec(rh, prec, rh_min, flag_val):
    r"""
    Check when relative humidity is below a certain value (rh_min), while precipitation
    is occurring.

    Parameters
    ----------
    rh : pd.DataFrame
            Relative humidity dataframe to be tested (time, stations_rh) in percentage
    prec : pd.DataFrame
            Precipitation dataframe to be tested (time, stations_prec) in mm
    rh_min: float
            Minimum acceptable value of relative humidity during a precipitation event.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying relative humidity values which fail the test.

    Warning
    -------
    This test is applied only to stations measuring both precipitation and
    relative humidity.
    """
    stat_rh_prec = rh.columns.intersection(prec.columns)

    flag = pd.DataFrame(0, index=rh.index, columns=rh.columns)

    mask_precrh = ((prec.loc[:, stat_rh_prec] > 0) & (rh.loc[:, stat_rh_prec] < rh_min))
    flag[mask_precrh] = flag_val

    return flag


def leafwet_humidity(lw, rh, rh_min, flag_val):
    r"""
    Check when relative humidity is below a certain value (rh_min), while leaf wetness
    duration is bigger than 0.

    Parameters
    ----------
    lw : pd.DataFrame
            Leaf wetness duration dataframe to be tested (time, stations_lw) in min
    rh : pd.DataFrame
            Relative humidity dataframe to be tested (time, stations_rh) in percentage
    rh_min: float
            Minimum acceptable value of relative humidity for observing leaf wetness duration.
    flag_val : int
            integer representing flag values to be associated to
            erroneous values.

    Returns
    -------
    flag : pd.DataFrame
            Dataframe with flags identifying leaf wetness values which fail the test.

    Warning
    -------
    This test is applied only to stations measuring both leaf wetness and
    relative humidity.
    """
    stat_lw_rh = lw.columns.intersection(rh.columns)

    flag = pd.DataFrame(0, index=lw.index, columns=lw.columns)

    mask_lwrh = ((lw.loc[:, stat_lw_rh] > 0) & (rh.loc[:, stat_lw_rh] < rh_min))
    flag[mask_lwrh] = flag_val

    return flag
