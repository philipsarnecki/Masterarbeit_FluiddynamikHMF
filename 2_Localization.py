import os
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
import pandas as pd
import pickle
import rasterio
from shapely.geometry import Point
import h5py
datapath = r"C:\Users\Philip\Desktop\MA\Data"
ArcGISpath  = r"C:\Users\Philip\Desktop\MA\MFP_ArcGIS"
scriptpath = r"C:\Users\Philip\Desktop\MA\Python&Matlab\Github"
os.chdir(scriptpath)
from Functions import *

os.chdir(datapath)

# load coordinates
COORDS_DATA = "koord_relativ.csv"
coords = np.loadtxt(COORDS_DATA)[:, :] # [E/x/long, N/y/lat]
rx = coords[:, 0]
ry = coords[:, 1]

# Velocity
Rayleigh = 443
Vp = 3460
Vs = 3460/100*60

# sampling frequency (200 samples per second)
sampl_freq = 200
# time range to compute [min]
min_compute = 540;
# total nr of samples
NSAMP = sampl_freq*60*min_compute

# create array with potential win sizes
pot_win_sizes = np.arange(5, min_compute)

threshold_percentage = 90
threshold_normalized = 0.9

with rasterio.open(os.path.join (ArcGISpath, "AS_28_modifiziert_relativ" + ".tif")) as r:
    xmin, xmax = r.bounds.left, r.bounds.right
    ymin, ymax = r.bounds.bottom, r.bounds.top
GRDX = 50 # number of grid points
GRDY = 50
xx, yy = np.meshgrid(np.linspace(xmin, xmax, GRDX), np.linspace(ymin, ymax, GRDY), indexing='ij')

#%% Settings

win_sizes = [30, 20, 15, 10, 5, 1]

components = ["Z", "E", "N"]
depths = [0, 100, 200, 300, 400, 500]

localization = 0
hdf_to_dict = 0

analysis_method = "fluid"
# "win", "comp", "fluid"

#%% MFP Localization

if localization == 1:

    for component in components:
        
        print ('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- Component:', component)
        
        # open/create hdf5 file for current component where MFP will be stored
        MFP = h5py.File("MFP_" + component + ".hdf5", 'a')
        
        init = h5py.File("init_" + component,'r+')
        init = np.array(init["init_" + component])
        
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
            
            print ("-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- Win Size:", win_size_key)
            
            try: # check if folder for current winsize already exists
                MFP[win_size_key]
            except:
                MFP.create_group(win_size_key)
            
            for depth in depths:
                
                depth_key = str(depth) + "m"
                
                print ("-x-x-x-x-x-x-x-x-x-x-x-x- Depth:", depth_key)
                
                # output depth & nr of layers
                depth_start = depth
                depth_end = depth_start
                nr_layers = 1
                # desired output depth with number of layers
                DEPTHS = np.linspace(depth_start, depth_end, nr_layers)
    
                if component == "z": 
                    VEL = Vp
                else:
                    VEL = Vs
                if depth == 0:
                    VEL = Rayleigh
    
                win_depth = win_size_key + "_" + depth_key
                
                try: # check if folder for current winsize and current depth already exists
                    MFP[win_size_key][win_depth]
                    print ("MFP already exists")
                except:
                    print ("MFP does not exist yet")
                    # create subfolder for current depth
                    MFP_ = MFP[win_size_key].create_group(win_depth)
        
                    MFP_process(component, init, NSAMP, win_size, xmin, ymin, GRDX, GRDY, xmax, ymax, DEPTHS, MFP_, rx, ry, VEL, depth_start, sampl_freq)
    MFP.close()
    
#%% CONVERT MFP-ARRAYS TO PICKLE

if hdf_to_dict == 1:
    
    MFP_all = {}
    
    for component in components:
        
        print ('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- Component:', component)
    
        MFP = h5py.File("MFP_" + component + ".hdf5", 'a')
        
        MFP_all[component] = {}
        
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
            
            MFP_all[component][win_size_key] = {}
            
            print ("-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- Win Size:", win_size_key)
            
            window_infos = []
            WINDOWS = int(sampl_freq*60*win_size)
            for a in range (0, NSAMP, WINDOWS):
                win_start = int(a / sampl_freq / 60)
                win_end = int((a+WINDOWS) / sampl_freq / 60)
                window_info = str(win_start).zfill(3) + "_to_" + str(win_end).zfill(3)
                window_infos.append(window_info)
                
            # get depth arrays for each window
            for window_info in window_infos:
                MFP_all[component][win_size_key][window_info] = {}
                for depth in depths:
                    MFP_all[component][win_size_key][window_info][depth] = {}
                    depth_key = str(depth) + "m"
                    win_depth = win_size_key + "_" + depth_key
                    MFP_all[component][win_size_key][window_info][depth] = np.array(MFP[win_size_key][win_depth][str(depth) + "m_" + window_info])
    
    with open("MFP_all.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_all, pickle_dict)
        pickle_dict.close()
    
#%% WINDOW-ANALYSIS

if analysis_method == "win":
    
    print ('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- WIN-ANALYSIS')

    ############ 1) Normalize each window for itself
    # -> i.e. normalize over all depths
    # -> each win size, all components
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_all.pickle", 'rb') as pickle_dict: 
        MFP_all = pickle.load(pickle_dict)
    
    MFP_all_norm = {}
    
    for component in MFP_all:
        
        MFP_all_norm[component] = {}
        
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
            
            MFP_all_norm[component][win_size_key] = {}
                
            # get arrays for each window and normalize
            for window_info in MFP_all[component][win_size_key]:
                MFP_all_norm[component][win_size_key][window_info] = {}
                
                global_max = max([array.max() for array in MFP_all[component][win_size_key][window_info].values()])
                global_min = min([array.min() for array in MFP_all[component][win_size_key][window_info].values()])
                norm_arrays = {key: Normalize(array, global_min, global_max) for key, array in MFP_all[component][win_size_key][window_info].items()}
                
                MFP_all_norm[component][win_size_key][window_info] = norm_arrays
                    
    with open("MFP_all_norm.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_all_norm, pickle_dict)
        pickle_dict.close()
    
    ############ 2) Get windows with normalized depth bmax > 0.9 and their percentage (all win sizes, all comp)
    # -> each depth normalized across all windows
    # -> i.e. in step 1 each window is normalized for itself (to get the true localization) 
    #         and now these normalized windows are again normalized, but now each depth for itself (across all windows)
    
    MFP_above_90 = {}
    percentages = {}
    
    for component in MFP_all_norm:
        
        MFP_above_90[component] = {}
        percentages[component] = {}
        percentages_df = pd.DataFrame(columns=[win_sizes])
    
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
            
            MFP_above_90[component][win_size_key] = {}
    
            # get window arrays of current win size (all depths)
            MFP_win = MFP_all_norm[component][win_size_key]
            
            percentages[component][win_size_key] = {}
            
            for depth in depths:
                
                MFP_above_90[component][win_size_key][depth] = {}
                
                # get window arrays of current depth
                MFP_depth = {win_info: MFP_win[win_info][depth] for win_info in MFP_win}
            
                # get global max & min (i.e. max & min across all window arrays of current depth)
                global_max = max([MFP_depth[array].max() for array in MFP_depth])
                global_min = min([MFP_depth[array].min() for array in MFP_depth])
                
                # normalize window arrays of current win size of current depth (normalize across all windows of current win size of current depth)
                MFP_norm = {key: Normalize(array, global_min, global_max) for key, array in MFP_depth.items()}
                
                # get arrays with normalized bmax above 90% for current depth
                MFP_above_90[component][win_size_key][depth] = {key: array for key, array in MFP_norm.items() if array.max() >= threshold_normalized}
                    
                # get amount of windows that has normalized bmax above 50%
                win_above = len(MFP_above_90[component][win_size_key][depth])
                
                # convert to percentage in relation to respective total number of windows
                win_above_perc = win_above / (540/win_size/100)
                
                percentages[component][win_size_key][depth] = win_above_perc
                
                # without window amounts
                # percentages_df.loc[depth, win_size] = (str(int(win_above_perc)) + " %")
                percentages_df.loc[depth, win_size] = (int(win_above_perc))
                
                # with window amounts
                # percentages_df.loc[depth, win_size] = (str(int(win_above_perc)) + " %" + " (" + str (win_above) + ")")
        
        percentages_df.style.background_gradient(cmap='PuBu', low=0, high=0, axis=1, subset=None).to_excel(component + '_percentages_above_thres.xlsx')
    
    with open("MFP_above_90.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_above_90, pickle_dict)
        pickle_dict.close()
            
    with open("percentages.pickle", 'wb') as pickle_dict:
        pickle.dump(percentages, pickle_dict)
        pickle_dict.close()
    
    ############ 3) Get horizontal resolution (i.e. averaged bp range over all windows for each depth) (all win sizes, all comp)
    
    h_res_dict = {}
    
    for component in MFP_all:
        
        h_res_dict[component] = {}
        
        h_res_df = pd.DataFrame(columns = [win_sizes])
        
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
                
            h_res_dict[component][win_size_key] = {}
            
            df = pd.DataFrame ((), columns = ["depth", "bp_range"])
            
            # get window arrays of current win size (all depths)
            MFP_win = MFP_all[component][win_size_key]
            
            for depth in depths:
                
                depth_key = str(depth) + "m"
                
                win_depth = win_size_key + "_" + depth_key
                
                # get window arrays of current depth
                MFP_depth = {win_info: MFP_win[win_info][depth] for win_info in MFP_win}
                
                for key, value in MFP_depth.items():
                    bp_range = np.ptp(MFP_depth[key])
                    df = df.append({'depth': int(depth), "bp_range": bp_range}, ignore_index = True)
            
            # calculate mean bp range for each depth and get min and max bp range of each depth
            # -> https://stackoverflow.com/questions/46501703/groupby-column-and-find-min-and-max-of-each-group
            df_ = (df.assign(Data_Value = df['bp_range'].abs()).groupby(['depth'])['bp_range'].agg([('bp_range_min', 'min'), ('bp_range_max', 'max'), ('bp_range_mean', 'mean')]))
            
            # calculate the difference between the mean bp range and the min and max bp range
            df_['diff_mean_max'] = df_['bp_range_max'] - df_['bp_range_mean']
            df_['diff_mean_min'] = df_['bp_range_mean'] - df_['bp_range_min']
                
            h_res_dict[component][win_size_key] = df_
            
            h_res_df[win_size] = df_.loc[:, 'bp_range_mean']
             
        h_res_df.style.background_gradient(cmap = 'PuBu', low = 0, high = 0, axis = 1, subset = None).to_excel(component + '_h_res.xlsx')
            
    with open("h_res.pickle", 'wb') as pickle_dict:
        pickle.dump(h_res_dict, pickle_dict)
        pickle_dict.close()
    
#%% COMPONENT-ANALYSIS

if analysis_method == "comp":
    
    print ('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- COMP-ANALYSIS')
    
    ############ Stack normalized windows
    # -> win size 1min, all components
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_all_norm.pickle", 'rb') as pickle_dict: 
        MFP_all_norm = pickle.load(pickle_dict)
    
    MFP_norm_stacked = {}
    MFP_norm_stacked_max = {}
    MFP_norm_stacked_max_df = pd.DataFrame(columns=[components])
    
    for component in components:
        
        MFP_norm_stacked[component] = {}
        
        # stack (i.e. sum) the normalized array values of each depth for current component (win size 1min)
        # https://stackoverflow.com/questions/42693487/sum-values-of-similar-keys-inside-two-nested-dictionary-in-python
        MFP_norm_stacked[component] = reduce(lambda x, y: dict((k, v + y[k]) for k, v in x.items()), MFP_all_norm[component]["1min"].values())
        MFP_norm_stacked_max[component] = {depth: array.max() for depth, array in MFP_norm_stacked[component].items()}
    
        df = pd.DataFrame.from_dict(MFP_norm_stacked_max[component], orient = 'index', columns = [component])
        MFP_norm_stacked_max_df[component] = df.loc[:, component]
         
    MFP_norm_stacked_max_df.style.background_gradient(cmap = 'PuBu', low = 0, high = 0, axis = 0, subset = None).to_excel("norm_stack.xlsx")
        
    with open("MFP_norm_stacked.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_norm_stacked, pickle_dict)
        pickle_dict.close()
        
    ############ Apply thres to normalized stacked arrays
    # -> win size 1min, all components
    # -> values below thres are converted to "0"
    
    MFP_stacked_thres = {}
    
    for component in components:
    
        global_max = max([array.max() for array in MFP_norm_stacked[component].values()])
        
        dummy_array = np.linspace (0, global_max, 100)
        # dummy_array = np.arange (1, global_max + 1, 100)
        threshold = np.percentile(dummy_array, threshold_percentage)
        
        MFP_stacked_thres[component] = {depth: np.where(array < threshold, 0, array) for depth, array in MFP_norm_stacked[component].items()}
    
    with open("MFP_stacked_thres.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_stacked_thres, pickle_dict)
        pickle_dict.close()
        
    ############ Get vertical resolutions
    # -> win size 1min, all components
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_all.pickle", 'rb') as pickle_dict: 
        MFP_all = pickle.load(pickle_dict)
    
    v_res_dict = {}
    
    for component in components:
        
        v_res_dict[component] = {}
    
        for win_size in win_sizes:
            
            win_size_key = str(win_size) + "min"
                
            v_res_dict[component][win_size_key] = {}
            
            df = pd.DataFrame ((), columns = ["depth", "bp_max"])
            
            # get window arrays of current win size (all depths)
            MFP_win = MFP_all[component][win_size_key]
            
            for depth in depths:
                
                depth_key = str(depth) + "m"
                
                win_depth = win_size_key + "_" + depth_key
                
                # get window arrays of current depth
                MFP_depth = {win_info: MFP_win[win_info][depth] for win_info in MFP_win}
                
                # get bp max of each window at store for current depth
                for key, value in MFP_depth.items():
                    bp_max = MFP_depth[key].max()
                    df = df.append({'depth': depth, "bp_max": bp_max}, ignore_index = True)
            
            # calculate mean bp max for each depth and get min and max bp max of each depth
            # -> https://stackoverflow.com/questions/46501703/groupby-column-and-find-min-and-max-of-each-group
            df_ = (df.assign(Data_Value = df['bp_max'].abs()).groupby(['depth'])['bp_max'].agg([('bp_max_min', 'min'), ('bp_max_max', 'max'), ('bp_max_mean', 'mean')]))
            
            # calculate the difference between the mean bp max and the min and max bp max
            df_['diff_mean_max'] = df_['bp_max_max'] - df_['bp_max_mean']
            df_['diff_mean_min'] = df_['bp_max_mean'] - df_['bp_max_min']
                
            v_res_dict[component][win_size_key] = df_
            
    with open("v_res.pickle", 'wb') as pickle_dict:
        pickle.dump(v_res_dict, pickle_dict)
        pickle_dict.close()
    
#%% FLUIDDYNAMIC-ANALYSIS

if analysis_method == "fluid":
    
    print ('-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x- FLUID-ANALYSIS')

    ############ Convert normalized stacked thres arrays to polygon
    # -> win size 1min, all components
        
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_stacked_thres.pickle", 'rb') as pickle_dict: 
        MFP_stacked_thres = pickle.load(pickle_dict)
        
    MFP_thres_poly = {}
        
    for component in components:
        MFP_thres_poly[component] = {}
        for depth, array in MFP_stacked_thres[component].items():
            if depth != 0:
                array = MFP_stacked_thres[component][depth]
                array = np.where(array != 0, 1, "nan")
                contf = plt.contourf(xx, yy, array)
                polys = ContourToPoly (contf, multipoly = "False")
                if polys:
                    # add signatures as keys (e.g. 100A)
                    polys_ = {str(depth) + chr(ord('A') + poly_idx): poly for poly_idx, poly in enumerate (polys)}
                    MFP_thres_poly[component][depth] = polys_
                    
    
    # Z: change signatures of polygons "A" to signatures "D" in depths 200 - 500
    for depth in np.arange(200, 600, 100):
        MFP_thres_poly["Z"][depth][str(depth) + "D"] = MFP_thres_poly["Z"][depth].pop(str(depth) + "A")
    # change signatures in depths 100
    MFP_thres_poly["Z"][100].pop(str(100) + "B")
    MFP_thres_poly["Z"][100][str(100) + "B"] = MFP_thres_poly["Z"][100].pop(str(100) + "A") # area with "A" will be renamed as "B"
    MFP_thres_poly["Z"][100][str(100) + "A"] = MFP_thres_poly["Z"][100].pop(str(100) + "D") # rename "D" temporarily as "A"
    MFP_thres_poly["Z"][100][str(100) + "D"] = MFP_thres_poly["Z"][100].pop(str(100) + "C")
    MFP_thres_poly["Z"][100][str(100) + "C"] = MFP_thres_poly["Z"][100].pop(str(100) + "A")
            
    # E: change signatures of polygons "B" to signatures "C" in depths 400 & 500
    for depth in np.arange(400, 600,100):
        MFP_thres_poly["E"][depth][str(depth) + "C"] = MFP_thres_poly["E"][depth].pop(str(depth) + "B")
    # change signatures of polygons "A" to signatures "B" in depths 200 - 500
    for depth in np.arange(200, 600,100):
        MFP_thres_poly["E"][depth][str(depth) + "B"] = MFP_thres_poly["E"][depth].pop(str(depth) + "A")
            
    with open("MFP_thres_poly.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_thres_poly, pickle_dict)
        pickle_dict.close()
        
    ############ Get maximum Beampower within Polygons
    # -> win size 1min, all components
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_thres_poly.pickle", 'rb') as pickle_dict: 
        MFP_thres_poly = pickle.load(pickle_dict)
        
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_all.pickle", 'rb') as pickle_dict: 
        MFP_all = pickle.load(pickle_dict)
        
    # convert all rasterpoints of the MFP raster into shapely points
    raster_x, raster_y = xx.flatten(), yy.flatten()
    raster_coords = np.vstack((raster_x, raster_y)).T
    raster_points = [Point(raster_coord[0], raster_coord[1]) for raster_coord in raster_coords]
            
    # get rasterpoints coords that are within each polygon (bool array i.e. True at indices that are within polygon)
    # -> i.e. the array indices (flattened) within each polygon
    MFP_poly_indices = {}
    for component in MFP_thres_poly:
        MFP_poly_indices[component] = {}
        for depth in MFP_thres_poly[component]:
            MFP_poly_indices[component][depth] = {poly_key: np.array([poly.contains(raster_point) for raster_point in raster_points]) for poly_key, poly in MFP_thres_poly[component][depth].items()}
    
    # get bmax of each polygon
    MFP_poly_bmax = {}
    for component in MFP_poly_indices:
        for depth in MFP_poly_indices[component]:
            # get window arrays of current depth
            MFP_depth = {win_info: MFP_all[component]["1min"][win_info][depth] for win_info in MFP_all[component]["1min"]}
            # flatten each window array and convert win info to int (e.g. first window will be "1")
            MFP_depth = {win_idx: MFP_depth[win_array].flatten() for win_idx, win_array in enumerate(MFP_depth)}
            # get the respective polygons indices for current component and current depth
            poly_indices = MFP_poly_indices[component][depth]
            # get bmax of each window array at each polygon
            for poly_sig, poly_ind in poly_indices.items():
                MFP_poly_bmax[poly_sig] = {win_idx: max(win_array[poly_ind]) for win_idx, win_array in MFP_depth.items()}
               
    # convert each polygon dict to array (first column = win_idx i.e. row_idx, second columns = bmax at respective win_idx)
    MFP_poly_bmax = {poly_sig: np.array(list(MFP_poly_bmax[poly_sig].items())) for poly_sig in MFP_poly_bmax}
               
    with open("MFP_poly_bmax.pickle", 'wb') as pickle_dict:
        pickle.dump(MFP_poly_bmax, pickle_dict)
        pickle_dict.close()