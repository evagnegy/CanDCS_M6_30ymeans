#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 18:14:43 2024

@author: evg000
"""

import xarray as xr
import glob,os
import pandas as pd
from xclim import ensembles as xens
import sys
sys.path.append(os.path.abspath('..'))
from filepaths import paths

  
SSPs=['ssp126','ssp245','ssp585']

ref_period = '1981-2010'
#time_periods = ['2021-2050', '2031-2060', '2041-2070', '2051-2080', '2061-2090', '2071-2100']
time_periods = ['1951-1980','1961-1990','1971-2000','1981-2010','1991-2020','2001-2030','2011-2040','2021-2050', '2031-2060', '2041-2070', '2051-2080', '2061-2090', '2071-2100']

gcms = ['ACCESS-CM2', 'ACCESS-ESM1-5','BCC-CSM2-MR', 'CMCC-ESM2', 'CNRM-CM6-1','CanESM5',
          'CNRM-ESM2-1', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-g3', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC-ES2L',
          'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
          'NorESM2-MM', 'TaiESM1', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'KACE-1-0-G']


temp_dir = paths.output_dir + 'temp/'


#%%

#open a random M6 file to get attributes to copy over
sample_path = paths.m6_filepath + "/CanESM5_pr_r1i1p2f1_ssp126_MBCn_5_year_files/"
sample_file = "pr_day_MBCn+PCIC-Blend_CanESM5_historical+ssp126_r1i1p2f1_gn_19500101-19551231.nc"
ds = xr.open_dataset(sample_path + sample_file,decode_times=False)

for ssp in SSPs:
    print(ssp)
    
    #get all the model files for each ssp (from the intermediate files)
    ssp_filelist = []
    for gcm in gcms:
        openfile_mean30y=f"{temp_dir}pr_seas_MBCn+PCIC-Blend_{gcm}_{ssp}_mean30y_wrt_{ref_period}.nc"
        ssp_filelist.append(openfile_mean30y)
        
    # read all the model files and get the ens 10,50,90 percentile, rounding to 2 decimals
    ens = xens.create_ensemble(ssp_filelist)            
    dsOut = xens.ensemble_percentiles(ens, values=[10,50,90], split=True).round(2)

    # copy all global attributes from sample M6 file
    dsOut.attrs = ds.attrs
    
    # remove the attrs starting with "driving" since this is for the ensemble, not one gcm
    keys_to_remove = [key for key in dsOut.attrs if key.startswith('driving_')]
    for key in keys_to_remove:
        del dsOut.attrs[key]
    
    # copy the variable attributes from the M6 raw file
    for v in list(dsOut.data_vars):
        dsOut[v].attrs = ds['pr'].attrs
    
    # add a description to each variable 
    dsOut['pr_p10'].attrs['description'] = '30 year mean seasonal total precipitation: delta compared to 1981-2010. 10th percentile of ensemble.'
    dsOut['pr_p50'].attrs['description'] = '30 year mean seasonal total precipitation: delta compared to 1981-2010. 50th percentile of ensemble.'
    dsOut['pr_p90'].attrs['description'] = '30 year mean seasonal total precipitation: delta compared to 1981-2010. 90th percentile of ensemble.'
    
    # create datetimes from the horizons (starting on first day of the 30 year period)
    # note that even though this is seasonal data, it always is Jan 1 of that 30 year period
    def parse_time_horizon(time_str):
        start_year = int(time_str[:4])
        return pd.Timestamp(f'{start_year}-01-01')
    time_dt = pd.Index([parse_time_horizon(th) for th in dsOut['time_horizon'].values])
    
    # add that time variable as a coordinate 
    dsOut = dsOut.assign_coords({"time": ("time", time_dt)})
    
    # Swap dimension 'time_horizon' with 'time' dimension for all variables
    for var in dsOut.data_vars:
        dsOut[var] = dsOut[var].swap_dims({'time_horizon': 'time'})
    
    # remove time_horizon as a dimension (since its a duplicate of time in terms of purpose)
    dsOut = dsOut.drop_vars('time_horizon')
    
    # add it (horizon) back as a coordinate (not a dimension) with the dimension of time 
    dsOut = dsOut.assign_coords(horizon=('time', time_periods))
    
    # rename the precip vars to be more descriptive
    dsOut = dsOut.rename_vars({'pr_p10': 'prcptot_delta_1981_2010_p10', 'pr_p50': 'prcptot_delta_1981_2010_p50', 'pr_p90': 'prcptot_delta_1981_2010_p90'})
    
    # compress and save as netcdf
    comp = dict(zlib=True, complevel=5)
    encoding = {var: comp for var in dsOut.data_vars}
    
    outfile = '30yAvg_prcptot_seas_MBCn+PCIC-Blend_ensemble-percentiles_historical+' + ssp + '_1950-2100_Abs_Ano.nc'
    dsOut.to_netcdf(paths.output_dir +  outfile, encoding=encoding, format="NETCDF4")





#%% sanity checks
 
ds_test = xr.open_dataset(paths.output_dir + outfile)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
import matplotlib as mpl

fig = plt.figure(figsize=(10,10),dpi=200)
ax = fig.add_subplot(1,1,1, projection=ccrs.RotatedPole(pole_latitude=42.5,pole_longitude=83))
 
data = ds_test.sel(time_horizon='2031-2060',season='DJF')['prcptot_delta_1981_2010_p50'].values
cmap = 'bwr_r'
vmin = -100
vmax = 100
extend = 'both'
plt.pcolormesh(ds_test.lon,ds_test.lat,data,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax)
#plt.scatter(lons,lats,c=precip,s=0.3,transform=ccrs.PlateCarree(),cmap=cmap,vmin=vmin,vmax=vmax)


states_provinces = cf.NaturalEarthFeature(category='cultural',name='admin_1_states_provinces_lines',scale='50m',facecolor='none')

ax.coastlines(resolution='50m')
ax.add_feature(cf.BORDERS)
ax.add_feature(states_provinces)

ax.set_extent([-135,-55,40,85],crs=ccrs.PlateCarree())
#ax.set_extent([-142,-133,55,65],crs=ccrs.PlateCarree())

cbar_ax = fig.add_axes([0.2,0.15,0.62,0.02])

fig.colorbar(mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax)),cax=cbar_ax,orientation='horizontal',extend=extend)
cbar_ax.tick_params(labelsize=16)

