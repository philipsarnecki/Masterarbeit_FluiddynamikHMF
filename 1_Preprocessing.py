from obspy import read, UTCDateTime
import os
import numpy as np
import h5py

######## 1. Preprocess data (filter, downsampling etc.) & cut data to DataCube-Timeframe (23.11 02:00 - 23.11 03:00)
######## 2. Export as np array

# For DataCubes:
# define input path
DC_input_path = r"D:\Uni - Leipzig\MA\Data\structured\DataCubes_structured"

# For SummitXOne:
# define input path
Summit_input_path = r"D:\Uni - Leipzig\MA\Data\structured\SummitXOne_structured"

# define output path
output_path = r"C:\Users\Philip\Desktop\MA\Data\preprocessed"
output_path_init = r"C:\Users\Philip\Desktop\MA\Data"

# define time to cut
start_day = 22; start_hour = 20; start_min = 0;
end_day   = 23; end_hour   = 1; end_min = 0;

start = UTCDateTime(2016,11,start_day,start_hour,start_min)
end = UTCDateTime(2016,11,end_day,end_hour,end_min)

#%% DATACUBES
# 1. Preprocess data

# change directory to DataCube structured data path
os.chdir(DC_output_path)

# get list containing all filenames from the folder
file_names = os.listdir(DC_input_path)

# extract the datacube names from the filenames & insert them into "datacube_names" 
datacube_names = [file_name[0:3] for file_name in file_names]

# remove duplicates (otherwise list would have each datacube thrice (one for each component))
datacube_names = list(set(datacube_names))

for datacube_name in datacube_names:
        
    # read files to be preprocessed
    st_z = read(datacube_name + "*Z.sac")
    # cut current trace
    st_z = st_z.trim(starttime = start, endtime = end)
    # Downsampling, conversion counts to units, detrend & filter current trace
    # st_z.decimate(factor = 4, strict_length=False) # downsampling by factor 4 (from 200Hz to 50Hz)
    st_z[0].data = st_z[0].data*0.2648 # counts to nm/s
    st_z.detrend(type='linear') # detrend
    st_z[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
        
    st_ns = read(datacube_name + "*NS.sac")
    st_ns = st_ns.trim(starttime = start, endtime = end)
    # st_ns.decimate(factor=4, strict_length=False)
    st_ns[0].data = st_ns[0].data*0.2648
    st_ns.detrend(type='linear')
    st_ns[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
        
    st_ew = read(datacube_name + "*EW.sac")
    st_ew = st_ew.trim(starttime = start, endtime = end)
    # st_ew.decimate(factor=4, strict_length=False)
    st_ew[0].data = st_ew[0].data*0.2648
    st_ew.detrend(type='linear')
    st_ew[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
    
    # change directory to final data output path
    os.chdir(output_path)
    
    # export merged data
    st_z[0].write(datacube_name + "_Z.sac")
    st_ns[0].write(datacube_name + "_NS.sac")
    st_ew[0].write(datacube_name + "_EW.sac")

    # change directory back to DataCube structured data path for next iteration
    os.chdir(DC_input_path)
        
#%% SummitXOne

# change directory to SummitXOne structured data path
os.chdir(Summit_input_path)

# get list containing all filenames from the folder
file_names = os.listdir(Summit_input_path)
   
# extract the station IDs from the filenames & insert them into "station_IDs" 
station_IDs = [file_name[0:3] for file_name in file_names]

for station_ID in station_IDs:
        
    # read files to be preprocessed
    st_z = read(station_ID + "*Z.sac")
    # # cut current trace
    st_z = st_z.trim(starttime = start, endtime = end)
    # Downsampling, conversion counts to units, detrend & filter current trace
    # st_z.decimate(factor = 5, strict_length=False) # downsampling by factor 5 (from 250Hz to 50Hz)
    st_z[0].data = st_z[0].data*3.571428571428572e+04 # counts to nm/s
    st_z.detrend(type='linear') # detrend
    st_z[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
        
    st_ns = read(station_ID + "*NS.sac")
    st_ns = st_ns.trim(starttime = start, endtime = end)
    st_ns.decimate(factor=5, strict_length=False)
    st_ns[0].data = st_ns[0].data*3.571428571428572e+04
    st_ns.detrend(type='linear')
    st_ns[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
        
    st_ew = read(station_ID + "*EW.sac")
    st_ew = st_ew.trim(starttime = start, endtime = end)
    st_ew.decimate(factor=5, strict_length=False)
    st_ew[0].data = st_ew[0].data*3.571428571428572e+04
    st_ew.detrend(type='linear')
    st_ew[0].filter('bandpass', freqmin=10, freqmax=20, corners=2)
    
    # change directory to final output path
    os.chdir(output_path)
    
    # export merged data
    st_z[0].write(station_ID  + "_Z.sac")
    st_ns[0].write(station_ID  + "_NS.sac")
    st_ew[0].write(station_ID + "_EW.sac")

    # change directory back to SummitXOne structured data path for next iteration
    os.chdir(Summit_input_path)

#%% 2. Unite stations by components (rows: samples, columns: geophones) & export as np array

# Z
# change directory to preprocessed output path
os.chdir(output_path)

# read files to be merged
st_z = read("*Z.sac")
# -> st_z will contain all z-component traces of all stations

# sort files by station name (so it will be the same order as in koord_relativ)
st_z.sort(['station'])

# create empty numpy list for each component
# -> amount of rows = number of samples (len(st[0]))
# -> amount of columns = number of stations (len(st))
data_all_z = np.zeros((len(st_z[0]),len(st_z)))

# transfer each trace from each stream into an empty list
for i, trace in enumerate (st_z):
    data_all_z[:,i] = st_z[i].data

# save data as np array
# np.save("init_z", data_all_z)

# save data as hdf5
os.chdir(output_path_init)
f = h5py.File('init_z', 'w')
f.create_dataset("init_z", data = data_all_z, compression="gzip")
f.close()

#%% NS

os.chdir(output_path)

st_ns = read("*NS.sac")

st_ns.sort(['station'])

data_all_ns = np.zeros((len(st_ns[0]),len(st_ns)))

for i, trace in enumerate (st_ns):
    data_all_ns[:,i] = st_ns[i].data
    
# save data as np array
# np.save("init_ns", data_all_ns)

# save data as hdf5
os.chdir(output_path_init)
f = h5py.File('init_ns', 'w')
f.create_dataset("init_ns", data = data_all_ns, compression="gzip")
f.close()

#%% EW

os.chdir(output_path)

st_ew = read("*EW.sac")

st_ew.sort(['station'])

data_all_ew = np.zeros((len(st_ew[0]),len(st_ew)))

for i, trace in enumerate (st_ew):
    data_all_ew[:,i] = st_ew[i].data

# save data as np array
# np.save("init_ew", data_all_ns)

# save data as hdf5
os.chdir(output_path_init)
f = h5py.File('init_ew', 'w')
f.create_dataset("init_ew", data = data_all_ew, compression="gzip")
f.close()