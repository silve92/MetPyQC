import pyqc as qc
import pandas as pd
import numpy as np
from metpy.units import units
import metpy.calc as mpcalc


temp=pd.read_csv('/home/meteo/Documenti/UMBRIA-QC/umbria_qc_post_proc/PyQC/test_data/temp_data_mod.csv', header=0, index_col=0, parse_dates=True)
print(temp)
#temp_pre=pd.read_csv('../pyqc/test_data/temp_data.csv', header=0, index_col=0, parse_dates=True)
#print(temp_pre)
#rh=pd.read_csv('/home/meteo/Documenti/UMBRIA-QC/umbria_qc_post_proc/PyQC/test_data/rh_data_mod.csv', header=0, index_col=0, parse_dates=True)
#print(rh)
#prec=pd.read_csv('/home/meteo/Documenti/UMBRIA-QC/umbria_qc_post_proc/PyQC/test_data/prec_data_mod.csv', header=0, index_col=0, parse_dates=True)
#print(prec)
lat_lon=pd.read_csv(
    '/home/meteo/Documenti/UMBRIA-QC/umbria_qc_post_proc/PyQC/test_data/station_lat_lon.csv',
header=0)
#print(lat_lon)
lat=lat_lon['latitude'].values
lon=lat_lon['longitude'].values


#print(temp.index.month)

#flag_range, res_range =qc.range.range_all(temp, -30, 50, 1)
#print(res_range)

#flag_seas, res_seas=qc.range_seas(temp, np.array([-30,-10,-30,-10]), np.array([20,30,50,25]), 1)

#print((res_seas==res_range).sum())

#flag_step, res_step = qc.temporal.step_all(temp, 6, 15, 32, 64)
#print(flag_step)
#print(res_step)

#flag_pers, res_pers = qc.persistence_noc(temp, 3, 1)

#flag_pers_var, res_pers_var = qc.persistence_var(temp, 12, 80, 'STD', 0.1, 1)
# print(flag_pers_var)
# print(res_pers_var)

#flag_maxmin, res_maxmin = qc.maxmin(temp, temp, temp_pre, 1)
#print(flag_maxmin)
#print(res_maxmin)

#flag_dew, res_dew, dew = qc.dewpoint_test(temp, rh, 1)
#dew_metpy=np.round(mpcalc.dewpoint_from_relative_humidity(temp.values*units('degC'), rh.values/100).m, 1)
#print(dew-dew_metpy)
#print(flag_dew)
#print(res_dew)

#ind_heater=[1,2]
#flag_prectg = qc.internal.heated_raingauge(prec, temp, ind_heater, 1)
#print(flag_prectg)

#flag_isol, res_isol = qc.temporal.isolated(prec, 4, 1)
#print(flag_isol)
#print(res_isol)

Rs_temp=0.3
min_neigh_tg=3
interval_tg=48
start='2019-06-01 00:00:00'
end='2019-06-01 05:00:00'
temp_est, std_temp_est, flag_spat, res_spat  =qc.spatial.hubbard_consistency(
                    lat, lon, temp, start_test=start, end_test=end,
                    n_max= 5, t_max=interval_tg,
                    search_radius=Rs_temp, min_neigh=min_neigh_tg, missing_perc=50,
                    f=4, flag_val=1)
#print(temp_est)
print(std_temp_est)
#print(flag_spat)
#print(res_spat)