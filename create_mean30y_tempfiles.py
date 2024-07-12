import xarray as xr
import glob,os
import pandas as pd
from netCDF4 import Dataset
import numpy as np
import sys
sys.path.append(os.path.abspath('..'))
from filepaths import paths


var='pr'   #read from raw M6 pr variable       

ssp = sys.argv[1] #allow for parallel runs, options ssp126, ssp245, ssp585

ref_period = '1981-2010'
time_periods = ['1951-1980','1961-1990','1971-2000','1981-2010','1991-2020','2001-2030','2011-2040','2021-2050', '2031-2060', '2041-2070', '2051-2080', '2061-2090', '2071-2100']
#time_periods = ['2021-2050', '2031-2060', '2041-2070', '2051-2080', '2061-2090', '2071-2100']

gcms = ['ACCESS-CM2', 'ACCESS-ESM1-5','BCC-CSM2-MR', 'CMCC-ESM2', 'CNRM-CM6-1','CanESM5',
          'CNRM-ESM2-1', 'EC-Earth3-Veg', 'EC-Earth3', 'FGOALS-g3', 'GFDL-ESM4',
          'INM-CM4-8', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KIOST-ESM', 'MIROC-ES2L',
          'MIROC6', 'MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'MRI-ESM2-0', 'NorESM2-LM',
          'NorESM2-MM', 'TaiESM1', 'HadGEM3-GC31-LL', 'UKESM1-0-LL', 'KACE-1-0-G']

# directory to save intermediate files
temp_dir = paths.output_dir + 'temp/'

# mask file for M6 data
mask_file = paths.mask + 'mask_grid.nc'
nc_mask = Dataset(mask_file,'r')
mask_arr = nc_mask.variables['valid_mask'][:]
mask_xr = xr.DataArray(mask_arr,dims=('lat','lon'))

# reformat the horizons
time_periods_dt = [tuple(map(str, yr.split('-'))) for yr in time_periods]
time_periods_dt = [(yr1, yr2) for yr1, yr2 in time_periods_dt]
base_periods_dt = tuple(map(str, ref_period.split('-')))

# used for reading in raw M6 output
def get_ripf(model):
    if model in ['CNRM-CM6-1','CNRM-ESM2-1',"MIROC-ES2L","UKESM1-0-LL"]:
        ripf = 'r1i1p1f2'
    elif model == "HadGEM3-GC31-LL":
        ripf = 'r1i1p1f3'
    elif model == "KACE-1-0-G":
        ripf = 'r2i1p1f1'
    elif model == "EC-Earth3":
        ripf = 'r4i1p1f1'
    elif model == 'CanESM5':
        ripf = 'r1i1p2f1'
    else:
        ripf = "r1i1p1f1"

    return(ripf)

#%%

for gcm in gcms:
    print(ssp + ": " + gcm)
    
    #intermediate file name
    savefile_mean30y=f"pr_seas_MBCn+PCIC-Blend_{gcm}_{ssp}_mean30y_wrt_{ref_period}.nc"

    #check if it already was created before creating it
    if not os.path.isfile(temp_dir + savefile_mean30y):

        # grab all files for this GCM (theyre split into 5 year chunks)
        model_path = f"{paths.m6_filepath}{gcm}_{var}_{get_ripf(gcm)}_{ssp}_MBCn_5_year_files/"
        all_model_files = os.listdir(model_path)
        all_model_files.sort()

        all_model_filepaths = []
        for file in all_model_files:
            #print(file)
            if not file.endswith(".nc"):
                continue
            if gcm == "MPI-ESM1-2-HR" and "MIROC" in file: #wrong file in this folder
                continue

            all_model_filepaths.append(model_path + file)

        # open all files for each gcm at once 
        ds = xr.open_mfdataset(all_model_filepaths,decode_times=False)
        time_var = xr.decode_cf(ds).time

        if gcm in ['HadGEM3-GC31-LL', 'UKESM1-0-LL', 'KACE-1-0-G']: #handles 360day calendars
            ds['time'] = time_var.convert_calendar("standard",use_cftime=True,align_on='year')
        else:
            ds['time'] = time_var.astype('datetime64[ns]')

        ds_var = ds[var].to_dataset().load()
        ds_var.attrs.update(ds.attrs)
        

        # turn daily data into seasonal - summing each season so its seasonal precip totals
        ds_seas = ds_var.resample(time="QS-DEC").sum(dim='time')
        
        # reapply the dataset mask; sum turns NaNs to 0
        ds_seas_m = ds_seas.where(mask_xr, np.nan)
        
        # split the seasons into separate variables
        ds_seas_all_DJF = ds_seas_m.sel(time=ds_seas_m.time.dt.season=='DJF')
        ds_seas_all_MAM = ds_seas_m.sel(time=ds_seas_m.time.dt.season=='MAM')
        ds_seas_all_JJA = ds_seas_m.sel(time=ds_seas_m.time.dt.season=='JJA')
        ds_seas_all_SON = ds_seas_m.sel(time=ds_seas_m.time.dt.season=='SON')
        
        
        DJF_temp, MAM_temp, JJA_temp, SON_temp = [],[],[],[]
        for yr_start, yr_end in time_periods_dt:
            
            #select data from each 30 year period 
            #DJF is offset bc of december being in prev year (QS-DEC gives datetimes of starting season date)
            DJF_temp.append(ds_seas_all_DJF.sel(time=slice(str(int(yr_start)-1),str(int(yr_end)-1))).mean(dim='time'))
            MAM_temp.append(ds_seas_all_MAM.sel(time=slice(yr_start,yr_end)).mean(dim='time'))
            JJA_temp.append(ds_seas_all_JJA.sel(time=slice(yr_start,yr_end)).mean(dim='time'))
            SON_temp.append(ds_seas_all_SON.sel(time=slice(yr_start,yr_end)).mean(dim='time'))
        
        # put all the 30yr means into one dataset with dimension of time_horizon
        DJF_30yr_means = xr.concat(DJF_temp, dim=pd.Index(time_periods, name='time_horizon'))
        MAM_30yr_means = xr.concat(MAM_temp, dim=pd.Index(time_periods, name='time_horizon'))
        JJA_30yr_means = xr.concat(JJA_temp, dim=pd.Index(time_periods, name='time_horizon'))
        SON_30yr_means = xr.concat(SON_temp, dim=pd.Index(time_periods, name='time_horizon'))
        
        # get the base period mean
        DJF_30yr_base_mean = ds_seas_all_DJF.sel(time=slice(str(int(base_periods_dt[0])-1),str(int(base_periods_dt[-1])-1))).mean(dim='time')
        MAM_30yr_base_mean = ds_seas_all_MAM.sel(time=slice(base_periods_dt[0],base_periods_dt[-1])).mean(dim='time')
        JJA_30yr_base_mean = ds_seas_all_JJA.sel(time=slice(base_periods_dt[0],base_periods_dt[-1])).mean(dim='time')
        SON_30yr_base_mean = ds_seas_all_SON.sel(time=slice(base_periods_dt[0],base_periods_dt[-1])).mean(dim='time')
        
        # subtract the base period from all time horizons
        DJF_30yr_means_wrt_base = DJF_30yr_means - DJF_30yr_base_mean
        MAM_30yr_means_wrt_base = MAM_30yr_means - MAM_30yr_base_mean
        JJA_30yr_means_wrt_base = JJA_30yr_means - JJA_30yr_base_mean
        SON_30yr_means_wrt_base = SON_30yr_means - SON_30yr_base_mean
        
        # put all seasons into one dataset
        all_seas_30yr_means_wrt_base = [DJF_30yr_means_wrt_base,MAM_30yr_means_wrt_base,JJA_30yr_means_wrt_base,SON_30yr_means_wrt_base]
        all_seas_30yr_means_wrt_base_ds = xr.concat(all_seas_30yr_means_wrt_base, dim=pd.Index(['DJF','MAM','JJA','SON'], name='season'))
        
        # save as an intermediate netcdf file
        savefile_mean30y=f"pr_seas_MBCn+PCIC-Blend_{gcm}_{ssp}_mean30y_wrt_{ref_period}.nc"
        all_seas_30yr_means_wrt_base_ds.to_netcdf(temp_dir + savefile_mean30y)

