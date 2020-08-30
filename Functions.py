import numpy as np
from shapely.geometry import Polygon, MultiPolygon
import numpy as np
import matplotlib
from matplotlib import cm
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import scipy as sp
from scipy.stats import gaussian_kde
from scipy.interpolate import griddata
from scipy import ndimage
from scipy.signal import convolve2d
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as patches
import math
import copy
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.pylab as py
import mpl_toolkits.mplot3d.art3d as art3d
plt.interactive(True);
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import datetime
from descartes.patch import PolygonPatch
import matplotlib.patheffects as PathEffects
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import h5py
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import scipy

def MFP_process (component, init, NSAMP, win_size, XINIT, YINIT, GRDX, GRDY, XEND, YEND, DEPTHS, MFP, rx, ry, VEL, depth_start, sampl_freq):
    
    # load init data
    xt_amp = init
    
    DT = 0.02# sampling rate/interval ΔT [sec] (nr of sec per sample)
    freq = np.linspace(10, 20, 11) # frequency band
    FAVE = len(freq) # frequencies to average over
    
    WINDOWS = int(sampl_freq*60*win_size)
    
    # indices that correspond to frequencies of interest
    # -> data with frequencies 10-20 Hz is picked
    FIND = np.round(freq * (WINDOWS * DT))
    FIND = FIND.astype(int)
  
    # create beam grid/search grid
    grdx, grdy = np.meshgrid(np.linspace(XINIT, XEND, GRDX), np.linspace(YINIT, YEND, GRDY), indexing='ij')
    
    ##### MFP TIME-LAPSE
    for a in range (0, NSAMP, WINDOWS):
        
        # extract current window from raw (unnormalized) data
        xt_extract = xt_amp [a:a+WINDOWS,:]
        
        # normalize the raw data (amplitude normalization (1 bit))
        maxamp = np.absolute(xt_extract)
        xt = xt_extract / (maxamp.max(axis=0))
        xt = xt.transpose()
        
        # amount of stations
        nr = xt.shape[0]
        
        # pick time window
        window_info = 'window [min]: %.0f to %.0f' % (a / sampl_freq / 60, (a+WINDOWS) / sampl_freq / 60)
        print(window_info)
        
        xt_tl = xt

        # FFT
        # create zero-array where the fft for every station will be stored 
        # (amount of lines = amount of stations, amount of rows = amount of frequencies of interest)
        ft = np.zeros((nr, int(WINDOWS // 2) + 1), dtype=np.complex)
        
        for h in range(nr):
        # calculate discrete fft for each station
            ft[h] = np.fft.rfft(xt_tl[h], WINDOWS)
            
        # CSDM
        def specMat(xt_tl,ft, nr, FAVE, FIND, WINDOWS):
            R = np.zeros((nr, nr), dtype=np.complex)
        
            for n in range(FIND[0], FIND[-1]):
                R += np.outer(ft[:, n], ft[:, n].conj()) 
            
            R /= (FIND[-1]-FIND[0])*(WINDOWS*0.5)**2 
            return R
        
        R = specMat(xt_tl,ft, nr, FAVE, FIND, WINDOWS)
        
        # perform singular value decomposition to CSDM matrix
        u, s, vT = np.linalg.svd(R)
        # chose only neig strongest eigenvectors
        u_m = u[:, :0]  # columns are eigenvectors 5
        v_m = vT[:0 :]  # rows (!) are eigenvectors 5
        # set-up projector
        proj = np.identity(R.shape[0]) - np.dot(u_m, v_m)
        # apply projector to data - project largest eigenvectors out of data
        ft_proj = np.dot(proj, ft)
        # calculate new csdm
        R = np.dot(ft_proj, ft_proj.conj().T)
        
        # calculate distance vectors between gridpoints & stations
        dist_per_depth = {}
        
        for ijk in DEPTHS:
            dist = np.zeros((nr, GRDX, GRDY)) 
            for i in range(nr):
                dist[i] = np.sqrt((grdx - rx[i]) ** 2 + (grdy - ry[i]) ** 2 + ijk ** 2)
            dist_per_depth[ijk] = dist
        
        # Beamformer
        results = np.zeros((len(DEPTHS), GRDX, GRDY))
        
        for _i, ijk in enumerate(DEPTHS):
            res = results[_i]
            for i in range(GRDX):
                for j in range(GRDY):
                    for n in range(FAVE):
                    
                        # simple replica vector
                        steer= np.exp(-1j*2*np.pi*freq[n]*dist[:,i,j]/VEL)

                        # Bartlett Processor
                        res[i, j] += (steer.T.conj().dot(R).dot(steer)).real / float(nr)
                        
            print('max is: %.02f dB at depth: %i' % (10 * np.log10(res.max()), ijk))
            # 10*log ist Definition von Dezibel (1 dezibel = 10*log der amplitude)

        res = 10 * np.log10(results)
        
        win_start = int(a / sampl_freq / 60)
        win_end = int((a+WINDOWS) / sampl_freq / 60)
        window_info = str(win_start).zfill(3) + "_to_" + str(win_end).zfill(3)
        
        # store MFP data for current window in hdf5 file
        MFP.create_dataset(str(depth_start) + "m_" + window_info, data = res[0], compression="gzip")


def ContourToPoly (contf, multipoly):
    ##### converts filled contours to shapely polygons
    ##### -> https://gis.stackexchange.com/questions/99917/converting-matplotlib-contour-objects-to-shapely-objects
    
    ##### contf: contourf handle
    ##### multipoly: set to "True" if contf should be converted to multipolygon

    polys = []
    for col in contf.collections:
        # Loop through all polygons that have the same intensity level
        for contour_path in col.get_paths(): 
            # Create the polygon for this intensity level
            # The first polygon in the path is the main one, the following ones are "holes"
            for ncp,cp in enumerate(contour_path.to_polygons()):
                x = cp[:,0]
                y = cp[:,1]
                new_shape = Polygon([(i[0], i[1]) for i in zip(x,y)])
                if ncp == 0:
                    poly = new_shape
                else:
                    # Remove the holes if there are any
                    poly = poly.difference(new_shape)
                    # Can also be left out if you want to include all rings
            polys.append(poly)
            
    if multipoly == "True":
        polys = MultiPolygon(polys) # creates one polygon, consisting of several polygons
            
    return(polys)

def Normalize (array, global_min, global_max):
    
    norm = (array-global_min)/(global_max-global_min)
    
    return norm
    
def PlotMFPLoczCount_2D (ax, array, global_bin_max, xx, yy, inset):
    ##### Plot MFP localizaton counts (i.e. overlap counts)
    
    #### array: sum of localization counts
    
    # set range and steps for colormap
    clevs = np.linspace(0, int(global_bin_max), int(global_bin_max) + 2)
    # Choose colormap
    cmap_coolwarm = pl.cm.gist_stern_r
    # Get the colormap colors
    my_cmap = cmap_coolwarm(np.arange(cmap_coolwarm.N))
    # Set alpha
    my_cmap[:,-1] = 1
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    # set color of values below "vmin" to "none"
    my_cmap.set_under("none")
    
    normalize = matplotlib.colors.Normalize(vmin = 1, vmax = global_bin_max, clip = True)
    
    contf = ax.contourf(xx, yy, array, levels = clevs, cmap = my_cmap, norm = normalize, extend = 'min', zorder = 1)
    
    ax.set_aspect(aspect = "equal")
    
    return contf

def PlotMFPLoczCountSubplot_3D (fig, ax, array, global_bin_max, xx, yy, component, depth, modus, unique_areas):
    ##### Plot MFP localizaton counts (i.e. overlap counts) for MFP 3D subplot

    ##### global_bin_max: overall max localization count (encompassing all win sizes)

    clevs = np.linspace(0, int(global_bin_max), int(global_bin_max) + 2)
    # Choose colormap
    cmap_coolwarm = pl.cm.gist_stern_r
    # Get the colormap colors
    my_cmap = cmap_coolwarm(np.arange(cmap_coolwarm.N))
    # Set alpha
    my_cmap[:,-1] = 1
    # Create new colormap
    my_cmap = ListedColormap(my_cmap)
    # set color of values below "vmin" to "gainsboro"
    my_cmap.set_under("white", alpha = 0.7)
    
    normalize = matplotlib.colors.Normalize(vmin = 1, vmax = global_bin_max, clip = True)
    if modus == "unique":
        contf = ax.contourf(xx, yy, array, levels = clevs, cmap = my_cmap, offset = depth, norm = normalize, extend = 'min', zdir = 'z', zorder = 0)
    else:
        contf = ax.contourf(xx, yy, array, levels = clevs, cmap = my_cmap, offset = depth, norm = normalize, extend = 'min', zdir = 'z', zorder = 0)

    if unique_areas != []:
        patch = PolygonPatch(unique_areas, facecolor = "none", edgecolor = "black", alpha = 1, zorder = 2000)
    return contf
    
    plt.close(fig)
    
    return patch

def MFP3D_norm_stack (MFP_norm_all, component, ax, fig, global_max, global_min, depths, xx, yy, xmin, ymin, xmax, ymax, mofettes):
    
    # plot surface for every depth
    for depth, array in MFP_norm_all.items():
        
        PlotMFPLoczCountSubplot_3D (fig, ax, array, global_max, xx, yy, component, depth, modus = [], unique_areas = [])
        
    ax.text(mofettes[0,0],mofettes[0,1], 0, u'\u2605', color = 'red', fontsize = 30, transform = ax.transData, zorder = 10000)
    ax.text(mofettes[1,0],mofettes[1,1], 0, u'\u2605', color = 'red', fontsize = 30, transform = ax.transData, zorder = 10000)
    ax.text(mofettes[2,0],mofettes[2,1], 0, u'\u2605', color = 'red', fontsize = 30, transform = ax.transData, zorder = 10000)
                        
    # set colours for colorbar
    clevs = np.arange(0, 1 + 0.1, step = 0.1)
    m = cm.ScalarMappable(cmap = cm.gist_stern_r)
    m.set_array(clevs)
    # set position [left, bottom, width, height]
    cax = fig.add_axes([0.95, 0.25, 0.07, 0.55])
    cb = fig.colorbar(m, cax = cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["%.f" % global_min, "%.f" % global_max])
    cax.tick_params(labelsize = 18)
    cb.set_label("Gestapelte norm. Beampower", labelpad = -15, size = 22)
                
    # draw north arrow              
    arrow3d(ax, length = 150, width = 2, head = 0.3, headwidth = 3, offset = [-200, 200, 500], theta_x = 270, theta_z = 0, color = "black")
    ax.text(-200, 300, 530, "N", size = 20, zorder = 1, color='k') 
    
    # set axis limits
    ax.set_zlim(min(depths),max(depths))
    ax.set_xlim(0,500)
    ax.set_ylim(0,500)
                
    # define distance titles to plots
    rcParams['axes.titlepad'] = -10
    
    # invert z- & x-axis
    ax.invert_zaxis()
    
    # set x-, y- & z-axislabel
    ax.set_xlabel('x [m]', labelpad = 0, fontsize = 20, verticalalignment = 'center')
    ax.set_ylabel('y [m]', labelpad = 3, fontsize = 20, verticalalignment = 'center')
    ax.set_zlabel('Tiefe [m]', labelpad = 20, fontsize = 22, rotation = 90)
    ax.zaxis.set_rotate_label(False) # disable automatic axislabel rotation
        
    # define ticklabels
    xy_labels = np.arange(xmin, xmax+1, 100)
    xy_labels = xy_labels.astype(int)
    z_labels = depths
    # set ticks and ticklabels
    # x
    ax.set_xticks(xy_labels) # create ticks
    ax.set_xticklabels(xy_labels, verticalalignment = 'baseline', horizontalalignment = 'center')
    ax.tick_params(axis='x', which='major', pad = 3, labelsize = 15)
    # y
    ax.set_yticks(xy_labels)
    ax.set_yticklabels(xy_labels, verticalalignment = 'bottom', horizontalalignment = 'right')
    ax.tick_params(axis='y', which='major', pad = 3, labelsize = 15)
    # z
    ax.set_zticks(z_labels)
    ax.set_zticklabels(z_labels, verticalalignment = 'center')
    ax.tick_params(axis = 'z', which = 'major', pad = 8, labelsize = 16)
    
    # rotate view
    ax.view_init(elev = 10, azim = 225)
    
def MFP3D_bmax_time_all_comp (fig, MFP_stacked_thres, MFP_thres_poly, MFP_poly_bmax, depths, xx, yy, xmin, ymin, xmax, ymax, mofettes, comp_colors):

    grid = fig.add_gridspec(nrows = 5, ncols = 7)

    ####################### Bmax vs time Plots
    
    # create dict containing the row index for each polygon
    # A: third column (because 3D plot will be in first&second column), B: fourth column etc.
    # Depth 100m: first row (i.e. row "0"), depth 200m: second row etc.
    # Example: 300B will be in third row (i.e. "2"), fourth column (i.e. "3")
    plot_axes = {}
    for row_idx, depth in enumerate(np.arange (100,600,100)): # 5 depths in total (100,200,300,400,500m)
        for column_idx, signature_idx in enumerate(np.arange (0,4,1)): # 4 signature letters in total (A,B,C,D)
            letter = chr(ord('A') + signature_idx)
            plot_axes[str(depth) + letter] = {}
            plot_axes[str(depth) + letter]["row"] = row_idx
            plot_axes[str(depth) + letter]["col"] = column_idx+2 # i add "2" to column idx because first&second column is for the 3d plot
    
    global_max = 79
    global_min = 69
        
        ###### Plot Bmax & Lok./Min.
    for poly_sig, poly_bmax in MFP_poly_bmax.items():
        
        row_idx = plot_axes[poly_sig]["row"]
        col_idx = plot_axes[poly_sig]["col"]
        ax = plt.subplot (grid[row_idx, col_idx])
        # ax2 = ax.twinx()
        
        # Get bmax data for current polygon
        x = MFP_poly_bmax[poly_sig][:,0]
        y = MFP_poly_bmax[poly_sig][:,1]
        # Sort the points by bmax, so that the lowest bmax are plotted last
        idx = y.argsort()
        x_, y_ = x[idx], y[idx]
        # Plot Bmax
        colormap = plt.cm.gist_stern_r
        normalize = matplotlib.colors.Normalize(vmin = y.min(), vmax = y.max())
        ax.scatter(x_, y_, c = y_, s = 80, cmap = colormap, norm = normalize, edgecolor = '')
        
        ###### Layout
        # general tick settings
        ax.tick_params(axis = 'both', which = 'major', labelsize = 16, length = 5, width = 1)
        
        ax.set_xlim(left = 0, right = 540)
        ax.set_ylim(top = global_max, bottom = global_min)
        
        ax.set_xticks(np.arange(0, 540 + 60, 60))
        ax.set_yticks(np.arange(global_min, global_max + 5, 5))
        
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, lw = 2, alpha = 0.5, zorder = -5)
        
        ticklabels = ax.get_xticklabels()
        
        ticklabels = ax.get_yticklabels()
        ticklabels[0].set_va("bottom")
        ticklabels[-1].set_va("top")

        ax.text(0, 1.02, poly_sig, fontsize = 19, weight = "bold", transform = ax.transAxes)
        
        # axis labels
        if poly_sig == "500B":
            ax.set_ylabel("Bmax [dB]", fontsize = 18, color = "black", weight = "bold")
            
        if poly_sig == "500B":
            ax.set_xlabel ("Uhrzeit", fontsize = 20, weight = "bold")
        
        # ticklabels
        labelbottom = ["100C", "400A", "500B", "500C", "500D"]
        if poly_sig in labelbottom:
            ax.tick_params(labelbottom = True)
            a = ax.get_xticks().tolist()
            for tickidx, tick in enumerate(a):
                a[tickidx] = ""
            
            a[0] = "22"
            a[1] = "23"
            a[2] = "0"
            a[3] = "1"
            a[4] = "2"
            a[5] = "3"
            a[6] = "4"
            a[7] = "5"
            a[8] = "6"
            a[9] = "7"
            
            ax.set_xticklabels(a)
        else:
            ax.tick_params(labelbottom = False)
        
        labelleft = ["100A", "200A", "300A", "400A", "500B"]
        if poly_sig in labelleft:
            ax.tick_params(labelleft = True)
        else:
            ax.tick_params(labelleft = False)
            
    ####################### 3D Plots
    ax_3d = plt.subplot (grid[0:, 0:2], projection='3d')
    
    # stretch z-axis (by 0.9)
    x_scale, y_scale, z_scale = 0.6, 0.6, 0.9
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    ax_3d.get_proj = lambda: np.dot(Axes3D.get_proj(ax_3d), np.diag([x_scale, y_scale, z_scale, 1]))

    # plot surface for every depth
    for component in MFP_stacked_thres:
        global_max = max([array.max() for array in MFP_stacked_thres[component].values()])
        for depth, array in MFP_stacked_thres[component].items():
            # add white rectangle as base for current depth
            rect = patches.Rectangle((0, 0), 500, 500, facecolor = (1, 1, 1, 0.5)) # color white (1, 1, 1) with alpha 0.3
            ax_3d.add_patch(rect)
            art3d.pathpatch_2d_to_3d(rect, z = depth, zdir = "z")
            array_ = np.where(array == 0, "nan", array)
            PlotMFPLoczCountSubplot_3D (fig, ax_3d, array_, global_max, xx, yy, component, depth, modus = [], unique_areas = [])
            # plot border of areas with color according to component
            array = np.where(array != 0, 1, array)
            ax_3d.contour(xx, yy, array, offset = depth, zdir = 'z', colors = comp_colors[component])
                    
    for component in MFP_thres_poly:
        for depth in MFP_thres_poly[component]:
            for poly_signature, poly in MFP_thres_poly[component][depth].items():
                poly_txt = ax_3d.text(poly.centroid.x + 30, poly.centroid.y, depth, poly_signature[-1:], color = 'white', fontsize = 32, weight = "bold", transform = ax_3d.transData, zorder = 1000000)
                poly_txt.set_path_effects([PathEffects.Stroke(linewidth = 2, foreground = "k"), PathEffects.Normal()])
    
    # plot mofettes
    ax_3d.text(mofettes[0,0],mofettes[0,1], 0, u'\u2605', color = 'red', fontsize = 40, transform = ax_3d.transData, zorder = 2000)
    ax_3d.text(mofettes[1,0],mofettes[1,1], 0, u'\u2605', color = 'red', fontsize = 40, transform = ax_3d.transData, zorder = 2000)
    ax_3d.text(mofettes[2,0],mofettes[2,1], 0, u'\u2605', color = 'red', fontsize = 40, transform = ax_3d.transData, zorder = 2000)
    
    # draw north arrow              
    arrow3d(ax_3d, length = 150, width = 2, head = 0.3, headwidth = 3, offset = [-200, 200, 500], theta_x = 270, theta_z = 0, color = "black")
    ax_3d.text(-200, 300, 530, "N", size = 24, zorder = 1, color='k')
    
    # set axis limits
    ax_3d.set_zlim(min(depths),max(depths))
    ax_3d.set_xlim(0,500)
    ax_3d.set_ylim(0,500)
                
    # define distance titles to plots
    rcParams['axes.titlepad'] = -10
    
    # invert z- & x-axis
    ax_3d.invert_zaxis()
    
    # set x-, y- & z-axislabel
    ax_3d.set_xlabel('x [m]', labelpad = 0, fontsize = 20, verticalalignment = 'center')
    ax_3d.set_ylabel('y [m]', labelpad = 1, fontsize = 20, verticalalignment = 'center')
    ax_3d.set_zlabel('Tiefe [m]', labelpad = 28, fontsize = 24, rotation = 90)
    ax_3d.zaxis.set_rotate_label(False) # disable automatic axislabel rotation
        
    # define ticklabels
    xy_labels = np.arange(xmin, xmax+1, 100)
    xy_labels = xy_labels.astype(int)
    z_labels = depths
    # set ticks and ticklabels
    # x
    ax_3d.set_xticks(xy_labels) # create ticks
    ax_3d.set_xticklabels(xy_labels, verticalalignment = 'baseline', horizontalalignment = 'center') # label ticks
    ax_3d.tick_params(axis = 'x', which = 'major', pad = 2, labelsize = 20) # adjust ticklabel positions
    # y
    ax_3d.set_yticks(xy_labels)
    ax_3d.set_yticklabels(xy_labels, verticalalignment = 'bottom', horizontalalignment = 'right')
    ax_3d.tick_params(axis = 'y', which = 'major', pad = 2, labelsize = 20)
    # z
    ax_3d.set_zticks(z_labels)
    ax_3d.set_zticklabels(z_labels, verticalalignment = 'center')
    ax_3d.tick_params(axis = 'z', which = 'major', pad = 12, labelsize = 20)
    
    # rotate view
    ax_3d.view_init(elev = 10, azim = 225)
    
    line_z = Line2D([0,1],[0,1],  linewidth = 8, linestyle = '-', color = comp_colors["Z"])
    line_ns = Line2D([0,1],[0,1], linewidth = 8, linestyle = '-', color = comp_colors["N"])
    line_ew = Line2D([0,1],[0,1], linewidth = 8, linestyle = '-', color = comp_colors["E"])
    plt.figlegend((line_z, line_ns, line_ew), ('Z', 'N', 'E'), loc = "center left", 
                  fontsize = 24, borderaxespad = 0.1, bbox_to_anchor = (0.23, 0.12))
    
    grid.update(wspace = 0.15)
    
    ax_3d.dist = 6
    
def percentages_above_thres (percentages, axes, win_sizes, win_size_key, win_size_idx, winsize_colors, component):
    
    if component == "Z":
        ax = axes[0]
    if component == "N":
        ax = axes[1]
    if component == "E":
        ax = axes[2]
    
    data = list(percentages[win_size_key].items())
    an_array = np.array(data)
    
    x = an_array[:,0]
    y = an_array[:,1]
    ax.plot(x, y, '--o', markersize = 10,  c = winsize_colors[win_size_idx])
        
    if component == "Z":
        global_min = 10
        global_max = 90
    if component == "N":
        global_min = 0
        global_max = 100
    if component == "E":
        global_min = 30
        global_max = 90
        
    ax.set_ylim([global_min, global_max])
        
    # make sure the ticks are set at every 20
    ax.set_yticks(np.arange(global_min, global_max + 20, 20))

    ax.axis(xmin = 0, xmax = 500)
    ax.grid(True)
    
    fontsize_lbls = 16
    
    if component == "N":
        ax.set_ylabel('Zeitfenster [%] mit\nnorm. Bmax > 0.9', fontsize = fontsize_lbls, weight = "bold")
    if component == "E":
        ax.set_xlabel('Tiefe [m]', fontsize = fontsize_lbls, weight = "bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(bottom = False, labelbottom = False)

    fontsize_ticks = 13
    ax.tick_params(axis = 'x', which = 'major', labelsize = fontsize_ticks)
    ax.tick_params(axis = 'y', which = 'major', labelsize = fontsize_ticks)
    
    # ax.set_title("Komponente: " + component, fontsize = 16, weight = "bold")
    ax.text(0, 1.02, component, fontsize = 22, weight = "bold", transform = ax.transAxes)
    
    if component == "N":
        legend_dict = {}
        for idx, win_size in enumerate (win_sizes):
            win_size_key = str(win_size) + " min."
            legend_dict[win_size_key] = winsize_colors[idx]
            
        LinesList = []
        win_sizes_vorlagen = ("1 min.", "5 min.", "10 min.", "15 min.", "20 min.", "30 min.") # make sure to sort legend in decreasing order
        for win_size_vorlage in win_sizes_vorlagen:
            for key in legend_dict:
                if key == win_size_vorlage:
                    data_key = Line2D([0,1],[0,1], marker = "o", linestyle = '--', color = legend_dict[key], label = key)
                    LinesList.append(data_key)
    
        ax.legend(handles = LinesList, loc = 'upper right', fontsize = 11, markerscale = 1, borderaxespad = 0.1)
        
def norm_stack_maxima (MFP_stacked_max, component, ax, fig, color):
    
    data = list(MFP_stacked_max.items())
    an_array = np.array(data)
    
    x = an_array[:,0]
    y = an_array[:,1]
    ax.plot(x, y, '--o', markersize = 10,  c = color)
    ax.axis(ymin = 310, ymax = 400)
    ax.axis(xmin = 0, xmax = 500)
    ax.grid(True)
    
    fontsize_lbls = 16
    ax.set_xlabel('Tiefe [m]', fontsize = fontsize_lbls, weight = "bold")
    ax.set_ylabel('Gestapelte norm. Beampower', fontsize = fontsize_lbls, weight = "bold")
        
    fontsize_ticks = 13
    ax.tick_params(axis = 'x', which = 'major', labelsize = fontsize_ticks)
    ax.tick_params(axis = 'y', which = 'major', labelsize = fontsize_ticks)
    
def MFP_Resolution_h_res (axes, h_res, component, win_sizes, winsize_colors, modus):
    # plot all win sizes in one plot, but without error bars

    if component == "Z":
        ax = axes[0]
    if component == "N":
        ax = axes[1]
    if component == "E":
        ax = axes[2]

    # Plot
    for i, (key, value) in enumerate (h_res.items()):
            h_res[key].plot(y = "bp_range_mean", marker = "o", linestyle = '--', ax = ax, color = winsize_colors[i], label = key, markersize = 12, grid = True, legend = False)
    
    fontsize_lbls = 16
    
    if component == "N":
        ax.set_ylabel('Ø Beampower-Spannbreite [dB]', fontsize = fontsize_lbls, weight = "bold")
    if component == "E":
        ax.set_xlabel('Tiefe [m]', fontsize = fontsize_lbls, weight = "bold")
    else:
        ax.set_xlabel("")
        ax.tick_params(bottom = False, labelbottom = False)

    fontsize_ticks = 13
    ax.tick_params(axis = 'x', which = 'major', labelsize = fontsize_ticks)
    ax.tick_params(axis = 'y', which = 'major', labelsize = fontsize_ticks)
    
    if component == "Z":
        global_min = 0.3
        global_max = 0.9
    if component == "N":
        global_min = 0.4
        global_max = 1
    if component == "E":
        global_min = 0.3
        global_max = 0.9
        
    ax.set_ylim([global_min, global_max])
        
    # make sure the ticks are set at every 0.1
    ax.set_yticks(np.arange(global_min, global_max + 0.1, 0.1))
    
    ax.text(0, 1.02, component, fontsize = 22, weight = "bold", transform = ax.transAxes)
    
    if component == "N":
        legend_dict = {}
        for idx, win_size in enumerate (win_sizes):
            win_size_key = str(win_size) + "min"
            legend_dict[win_size_key] = winsize_colors[idx]
            
        LinesList = []
        win_sizes_vorlagen = ("1min", "5min", "10min", "15min", "20min", "30min") # make sure to sort legend in decreasing order
        for win_size_vorlage in win_sizes_vorlagen:
            for key in legend_dict:
                if key == win_size_vorlage:
                    data_key = Line2D([0,1],[0,1], marker = "o", linestyle = '--', color = legend_dict[key], label = key)
                    LinesList.append(data_key)
    
        ax.legend(handles = LinesList, loc = 'upper right', fontsize = 11, markerscale = 1, borderaxespad = 0.1)
        
def MFP2D_norm_stack_0m (MFP_norm_stacked, component, axes, fig, global_max, global_min, xx, yy, xmin, ymin, xmax, ymax, mofettes):

    if component == "Z":
        ax = axes[0]
    if component == "N":
        ax = axes[1]
    if component == "E":
        ax = axes[2]

    PlotMFPLoczCount_2D (ax, MFP_norm_stacked, global_max, xx, yy, inset = [])
                    
    ax.text(mofettes[0,0],mofettes[0,1], u'\u2605', color = 'red', fontsize = 22, transform = ax.transData, zorder = 10000)
    ax.text(mofettes[1,0],mofettes[1,1], u'\u2605', color = 'red', fontsize = 22, transform = ax.transData, zorder = 10000)
    ax.text(mofettes[2,0],mofettes[2,1], u'\u2605', color = 'red', fontsize = 22, transform = ax.transData, zorder = 10000)
    
    # set colours for colorbar
    clevs = np.arange(0, 1 + 0.1, step = 0.1)
    m = cm.ScalarMappable(cmap = cm.gist_stern_r)
    m.set_array(clevs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size = '10%', pad = 0.05)
    cb = fig.colorbar(m, cax = cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["%.f" % global_min, "%.f" % global_max])
    cax.tick_params(labelsize = 12)
    if component == "N":
        cb.set_label("Gestapelte\n norm. Beampower", labelpad = -20, size = 14)

    # draw north arrow
    north = plt.imread(r"C:\Users\Philip\Desktop\MA\Karten\north_arrow.png")
    ax_n = inset_axes(ax, height = "12%", width = "12%", bbox_to_anchor = (-0.86, 0, 1, 1), bbox_transform = ax.transAxes)
    ax_n.imshow(north)
    ax_n.axis('off')
    
    # set axis limits
    ax.set_xlim(0,500)
    ax.set_ylim(0,500)
                
    fontsize_labels = 14
    fontsize_ticks = 12
    xy_ticklabels = np.arange(xmin, xmax+1, 100)
    xy_ticklabels = xy_ticklabels.astype(int)
    
    if component == "E":
        ax.set_xlabel('x [m]', fontsize = fontsize_labels)
        ax.set_xticks(xy_ticklabels)
        ax.tick_params(axis = 'x', which = 'major', labelsize = fontsize_ticks)
    else:
        ax.tick_params(labelbottom = False)
        
    ax.set_ylabel('y [m]', fontsize = fontsize_labels)
    ax.set_yticks(xy_ticklabels)
    ax.tick_params(axis = 'y', which = 'major', labelsize = fontsize_ticks)
    
    ax.text(0, 1.01, component, fontsize = 18, weight = "bold", transform = ax.transAxes)