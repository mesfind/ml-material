"""
This Python script converts GHCN station data to netCDF4 files, processing 50 stations per increment.

Parameters:
- nmin: Minimum number of observations required
- eles: List of elements to process
- start_date: Start date for data retrieval
- end_date: End date for data retrieval

Returns:
- NetCDF files containing station data for each element

Dependencies:
- xarray, pandas, datetime from datetime, esd

Usage:
- Ensure the required libraries are installed
- Run the script to generate netCDF files for GHCN station data

"""

import os
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
from esd import select_station, ele2param

nmin = 75
eles = [111, 121, 601]
start_date = datetime(1893, 1, 1)
end_date = datetime(2017, 12, 1)
date_range = pd.date_range(start_date, end_date, freq='D')

for ele in eles:
    ss = select_station(src='ghcnd', nmin=nmin, ele=ele)
    station_ids = ss['station_id'].tolist()
    num_stations = len(station_ids)
    print(f'Put {num_stations} stations world-wide in netCDF file')
    
    param = ele2param(ele, src='ghcnd')[4].lower()
    fname = f'{param}.ghcnd.nc'
    
    append = os.path.exists(fname)
    
    for i in range(0, num_stations, 50):
        station_subset = station_ids[i:i+50]
        print('Read data')
        print(i)
        
        try:
            station_data = ss.loc[ss['station_id'].isin(station_subset)]
            if (station_data.min().min() < -999) or (station_data.max().max() > 2000):
                print("Detected suspect data")
                print(station_data.min().min(), station_data.max().max())
                station_data = station_data.where((station_data >= -999) & (station_data <= 2000), other=pd.NA)
            
            ds = xr.Dataset({param: (['station', 'time'], station_data.values)},
                           coords={'station': station_subset, 'time': date_range})
            
            if append:
                ds.to_netcdf(fname, mode='a', unlimited_dims=['station'])
            else:
                ds.to_netcdf(fname)
                append = True
            
            print('added to netCDF file')
        
        except Exception as e:
            print('Failed to get data from GHCN:')
            print(param)
            print(station_subset)
            print(e)