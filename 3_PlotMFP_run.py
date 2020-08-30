import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import datetime
import rasterio
import os
import pickle
from matplotlib.lines import Line2D
import matplotlib.gridspec as gridspec
scriptpath = r"C:\Users\Philip\Desktop\MA\Python&Matlab\Analysis"
os.chdir(scriptpath)
from Functions import *

datapath    = r"C:\Users\Philip\Desktop\MA\Data" 
ArcGISpath  = r"C:\Users\Philip\Desktop\MA\MFP_ArcGIS"
COORDS_DATA = r"C:\Users\Philip\Desktop\MA\Data\koord_relativ.csv" # [E/x/long, N/y/lat]
outputpath  = r"C:\Users\Philip\Desktop\MA\Karten"

mofettes = np.genfromtxt(r"C:\Users\Philip\Desktop\MA\Data\Mofetten_relativ" +  ".csv")
geophones = np.genfromtxt(r"C:\Users\Philip\Desktop\MA\Data\koord_relativ" +  ".csv")

threshold = 0.75

# beam grid/search grid dimensions
with rasterio.open(os.path.join (ArcGISpath, "AS_28_modifiziert_relativ" + ".tif")) as r:
    xmin, xmax = r.bounds.left, r.bounds.right
    ymin, ymax = r.bounds.bottom, r.bounds.top
GRDX = 50 # grid spacing i.e. number of grid points -> int(xmax-xmin)
GRDY = 50 
xx, yy = np.meshgrid(np.linspace(xmin, xmax, GRDX), np.linspace(ymin, ymax, GRDY), indexing='ij')

title_size = 14
lable_size = 13
starttime = datetime.datetime(2016,11,23,22,00,00)

coords = np.loadtxt(COORDS_DATA)[:, :] # [E/x/long, N/y/lat]
rx, ry = coords[:, 0], coords[:, 1]

min_compute = 540

# sampling frequency
sampl_freq = 200
# total nr of samples for 1h
NSAMP = sampl_freq*60*min_compute

comp_colors = {"Z": "orange", "N": "blue", "E": "green"}

#%% Settings

win_sizes = [30, 20, 15, 10, 5, 1]
win_size = 1

components = ("Z", "N", "E")
component = "E"

winsize_colors = [cm.coolwarm_r(x) for x in np.linspace(0, 1, len(win_sizes))]

depths = [0, 100, 200, 300, 400, 500]

# define Modus for plotting
modus = "norm_stack"

#%% 3D MFP
    
### COMP-ANALYSIS: norm_stack
if modus == "norm_stack":
    # stack normalized beampower of all windows (one component)
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_norm_stacked.pickle", 'rb') as pickle_dict: 
        MFP_norm_stacked = pickle.load(pickle_dict)
    
    global_max = max([array.max() for array in MFP_norm_stacked[component].values()])
    global_min = min([array.min() for array in MFP_norm_stacked[component].values()])
    
    fig = plt.figure(figsize = (8, 12), dpi = 200); # width, height
    ax = plt.subplot (projection = '3d')

    MFP3D_norm_stack (MFP_norm_stacked[component], component, ax, fig, global_max, global_min, depths, xx, yy, xmin, ymin, xmax, ymax, mofettes)
    
    fig.suptitle(component, weight = "bold", fontsize = 24, y = 0.87, x = 0.51)
    plt.tight_layout()
    file_name = component + "_" + str(max(depths)) + "m_" + str(win_size) + "min_3D_" + modus + ".png"
    fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight")
    
### FLUIDDYNAMICS

if modus == "fluiddynamics":
    
    fig = plt.figure(figsize = (30, 12), dpi = 200) # width, height
        
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_stacked_thres.pickle", 'rb') as pickle_dict: 
        MFP_stacked_thres = pickle.load(pickle_dict)
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_thres_poly.pickle", 'rb') as pickle_dict: 
        MFP_thres_poly = pickle.load(pickle_dict)
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_poly_bmax.pickle", 'rb') as pickle_dict: 
        MFP_poly_bmax = pickle.load(pickle_dict)

    MFP3D_bmax_time_all_comp (fig, MFP_stacked_thres, MFP_thres_poly, MFP_poly_bmax, depths, xx, yy, xmin, ymin, xmax, ymax, mofettes, comp_colors)
    
    plt.tight_layout()
    file_name = str(max(depths)) + "m_" + modus + "_3D.png"
    fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight", pad_inches = 0.45)
    
#%% WIN-ANALYSIS: percentages_above_thres

# plot percentages of windows with norm bmax > 0.5 for every win size (only z)
# -> each depth normalized across all windows (after each window is normalized for itself)
if modus == "percentages_above_thres":
    # x axis: depths, y: axis percentage of windows with norm. bmax > 0.5 (only z)
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\percentages.pickle", 'rb') as pickle_dict: 
        percentages = pickle.load(pickle_dict)
        
        fig, axes = plt.subplots(figsize = (7, 8.5), dpi = 200, nrows = 3, ncols = 1) # width, height
        
        for component in components:
        
            for win_size_idx, win_size in enumerate(win_sizes):
                
                win_size_key = str(win_size) + "min"
    
                percentages_above_thres (percentages[component], axes, win_sizes, win_size_key, win_size_idx, winsize_colors, component)

        plt.tight_layout()
        plt.subplots_adjust(wspace = 0.1, hspace = 0.15)
        file_name = modus + ".png"
        fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight")
        
#%% COMP-ANALYSIS: norm_stack_maxima
# x axis: depths, y: norm stack maximum for each depth (all components in one plot)

if modus == "norm_stack_maxima":
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_norm_stacked.pickle", 'rb') as pickle_dict: 
        MFP_norm_stacked = pickle.load(pickle_dict)
    
    fig, ax = plt.subplots(figsize = (8, 6), dpi = 200) # width, height
    
    for component in components:
        
        # get max of the stacked arrays for each depth
        MFP_stacked_max = {depth: array.max() for depth, array in MFP_norm_stacked[component].items()}
        norm_stack_maxima (MFP_stacked_max, component, ax, fig, comp_colors[component])
        
    line_z = Line2D([0,1],[0,1],  linewidth = 6, linestyle = '-', color = comp_colors["Z"])
    line_ns = Line2D([0,1],[0,1], linewidth = 6, linestyle = '-', color = comp_colors["N"])
    line_ew = Line2D([0,1],[0,1], linewidth = 6, linestyle = '-', color = comp_colors["E"])
    plt.figlegend((line_z, line_ns, line_ew), ('Z', 'N', 'E'), loc = "center left", 
                  fontsize = 18, borderaxespad = 0.1, bbox_to_anchor = (0.8, 0.25))
        
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0, hspace = 0.15)
    file_name = modus + ".png"
    fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight")
     
#%% WIN-ANALYSIS: h_res
# -> plot h res (all win sizes)

if modus == "h_res":
    
    grid = gridspec.GridSpec(nrows = 2, ncols = 3, wspace = 0.05, hspace = 0.07)
    file_name = component + "_" + modus + ".png"
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\h_res.pickle", 'rb') as pickle_dict: 
        h_res = pickle.load(pickle_dict)
        
    fig, axes = plt.subplots(figsize = (7, 9), dpi = 200, nrows = 3, ncols = 1) # width, height
        
    for component in components:
    
        MFP_Resolution_h_res (axes, h_res[component], component, win_sizes, winsize_colors, modus)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0.1, hspace = 0.15)
    file_name = modus + ".png"
    fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight")
    
plt.close()
    
#%% COMP-ANALYSIS: norm_stack_0m
# -> plot stacked beampower for 0m (all components in one plot)

if modus == "norm_stack_0m":
    
    with open(r"C:\Users\Philip\Desktop\MA\Data\MFP_norm_stacked.pickle", 'rb') as pickle_dict: 
        MFP_norm_stacked = pickle.load(pickle_dict)
    
    fig, axes = plt.subplots(figsize = (7, 8.5), dpi = 200, nrows = 3, ncols = 1) # width, height
    
    for component in components:
        
        # get global min and max over all depths
        global_max = max([array.max() for array in MFP_norm_stacked[component].values()])
        global_min = min([array.min() for array in MFP_norm_stacked[component].values()])
    
        MFP2D_norm_stack_0m (MFP_norm_stacked[component][0], component, axes, fig, global_max, global_min, xx, yy, xmin, ymin, xmax, ymax, mofettes)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace = 0, hspace = 0.15)
    file_name = modus + ".png"
    fig.savefig(os.path.join (outputpath, file_name), bbox_inches = "tight")
    
plt.close()