'''
* Utility functions and classes to use for evaluating halo properties.
'''
import os
import sys
import struct
import binascii
import pandas as pd
import numpy as np
import h5py
from scipy.interpolate import interp1d
import time
#from scipy.integrate import quad
#import scipy.special as special
import pathlib
#import matplotlib
#import pickle
#from matplotlib import pyplot as plt
#import matplotlib.image as mpimg
#from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
#from mpl_toolkits import mplot3d
#from scipy.optimize import curve_fit
#from astropy.cosmology import FlatLambdaCDM
import importlib
#from IPython.display import display
#sys.path.insert(0, os.path.abspath('/Users/hk9457/Desktop/Hyunsu/Research/scripts'))
#sys.path.insert(0, os.path.abspath('/Users/hyunsukong/Desktop/Hyunsu/research/scripts'))
import rockstar_handling as rh
#
omega_l=0.6879
omega_m=0.3121
h=0.6751
H0=0.1*h  # km / s / kpc
part_mass=1.9693723*1e4/h
G=4.30091*1e-6 # kpc * Msolar^-1 * (km/s)^2
pd.set_option('display.max_columns', None)
#cosmo = FlatLambdaCDM(H0=67.51, Om0=0.3121, Tcmb0=2.725)

def distance_interpolation(sub_coord_arr, host_coord_arr, scale_arr, interp_factor):
    '''
    * This function interpolates distances between snapshots for an array of coordinates given. It uses an array interpolation instead of interpolating the orbit in a more physical way (e.g., using velocities/potential). It may not be the most accurate way, but it still is better than using just the snapshot data.
    * In an older version, it used scipy.interpolation.interp1d with kind='linear', but this updated version will use numpy.interp as scipy's interp1d is now a legacy class.
      - From my test with 20 cases, I find that the two methods give identical results.
    * This interpolates the x, y, and z coordinates of the subhalo and the host halo separately, then computes the distance.
    * scale_arr must be monotonically increasing for numpy.interp to work.
    * Returns:
      - interp_dist_arr
      - interp_scale_arr
    '''
    # Length of the new interpolated array I want.
    npts = int(len(sub_coord_arr) * interp_factor)
    # New scale factor array with finer time steps.
    interp_scale_arr = np.linspace(scale_arr[0], scale_arr[-1], npts)
    # Compute dx, dy, dz: component-wise distance from the center of the host halo for each particle.
    # Subhalo coordinate interpolation
    x_sub_interp = np.interp(interp_scale_arr, scale_arr, sub_coord_arr[:,0])
    y_sub_interp = np.interp(interp_scale_arr, scale_arr, sub_coord_arr[:,1])
    z_sub_interp = np.interp(interp_scale_arr, scale_arr, sub_coord_arr[:,2])
    # Host halo coordinate interpolation
    x_host_interp = np.interp(interp_scale_arr, scale_arr, host_coord_arr[:,0])
    y_host_interp = np.interp(interp_scale_arr, scale_arr, host_coord_arr[:,1])
    z_host_interp = np.interp(interp_scale_arr, scale_arr, host_coord_arr[:,2])
    # Compute the subhalo distance using the interpolated coordinates.
    interp_dist_arr = np.sqrt(
        (x_host_interp - x_sub_interp)**2. +
        (y_host_interp - y_sub_interp)**2. +
        (z_host_interp - z_sub_interp)**2.
    )
    return(interp_dist_arr, interp_scale_arr)
'''
* This function computes the circular velocity profile of a given halo at the given snapshot.
* Input: 
  - com: COM of the halo
  - coords: coordinates of halo particles
* Output:
  - vcirc_arr: an array of computed circular velocity values for particles sorted from innermost to outermost
  - sorted_part_dist: particles' distance sorted from innermost to outermost
  - sort_idx = np.argsort() indices that give sorted_part_dist
'''
def vcirc_particle_single_halo(com, coords):
    # Compute the distance for each particle
    part_dist = np.linalg.norm(coords - com, None, 1)
    
    # Sort the distance array
    sort_idx = np.argsort(part_dist)
    sorted_part_dist = part_dist[sort_idx]
    
    # Create an array with particle counts - 1, 2, 3, ...
    part_num_arr = np.arange(1, len(sorted_part_dist) + 1)
    
    # Compute Vcirc
    vcirc_arr = np.sqrt(G * part_mass * part_num_arr / sorted_part_dist)    
    '''
    vcirc_arr = np.zeros(len(sorted_part_dist))
    for i in range(len(vcirc_arr)):
        vcirc_arr[i] = np.sqrt(G * part_mass*(i+1)/sorted_part_dist[i])
    '''
    return(vcirc_arr, sorted_part_dist, sort_idx)

def vcirc_particle(com_list, coords_list, scale_list):
    vcirc_list = []
    dist_list = []
    
    if len(scale_list) == 0:
        # Scale factors are not passed.
        # It could be for z=0, or I could have forgotten to use it.
        
        for i in range(len(coords_list)):
            com = com_list[i]
            coords = coords_list[i]
        
            # Compute the distance for each particle
            part_dist = np.linalg.norm(coords - com, None, 1)
    
            # Sort the distance array
            sorted_part_dist = np.sort(part_dist)
    
            # Compute Vcirc
            vcirc_arr = np.zeros(len(sorted_part_dist))
            for i in range(len(vcirc_arr)):
                vcirc_arr[i] = np.sqrt(G * part_mass*(i+1)/sorted_part_dist[i])
            
            vcirc_list.append(vcirc_arr)
            dist_list.append(sorted_part_dist)
            
    else:
        # Use scale factors to convert to physical units.
        for i in range(len(coords_list)):
            com = com_list[i]
            coords = coords_list[i] * scale_list[i]
        
            # Compute the distance for each particle
            part_dist = np.linalg.norm(coords - com, None, 1)
    
            # Sort the distance array
            sorted_part_dist = np.sort(part_dist)
    
            # Compute Vcirc
            vcirc_arr = np.zeros(len(sorted_part_dist))
            for j in range(len(vcirc_arr)):
                vcirc_arr[j] = np.sqrt(G * part_mass*(j+1)/sorted_part_dist[j])
            
            vcirc_list.append(vcirc_arr)
            dist_list.append(sorted_part_dist)
    
    return(vcirc_list, dist_list)  

def get_rmax_col_from_ascii(catalog_path, snapshot_num, halo_id):
    # Create .ascii file names
    file_names=rh.get_file_name(catalog_path, snapshot_num, 'ascii')
    
    for file_name in file_names:

        # Open file
        text_df=pd.read_csv(file_name, sep=" ", low_memory=False)
    
        # Take the data part
        reduced_data=text_df[19:]
        # I just need two columns: id and num_p
        two_columns=reduced_data[['#id', 'rvmax']]
        two_columns=two_columns.rename(columns={'#id':'id'})
        two_columns.id=two_columns.id.astype(int)
        #two_columns.rvmax=two_columns.rvmax.astype(float)
        
        print(reduced_data[7162:7165])
    
        if halo_id in two_columns.id.values:
            # Add an index column and an old index column
            # old_index contains the original indices
            two_columns=two_columns.reset_index()
            two_columns.rename(columns={'index':'old_index'})
            two_columns=two_columns.reset_index()
            
            # Make .bin file name using the .ascii file name used
            bin_file_name=f'{file_name[:-5]}bin'
            
            #print(f'Name of .ascii file used: \n{file_name}')
            #print()
            return(two_columns, bin_file_name)
        
        else:
            continue

def halo_dist_below(dist_array, dist):
    for i, halo_dist in enumerate(dist_array):
        if halo_dist < dist:
            return i, halo_dist
        
# Get the scale factor at which the first infall of a halo occurs.
def find_first_infall_old(halo_df, host_rvir, vmax_min, vmax_max):
    num_elem = len(halo_df)
    for i in range(num_elem):
        if (np.flip(halo_df.dist.values)[i] < np.flip(host_rvir)[i]) and (np.flip(halo_df.vmax.values)[i] >= vmax_min) and (np.flip(halo_df.vmax.values)[i] <= vmax_max):
            return(np.flip(halo_df.scale.values)[i-1])        
        
# Get the scale factor at which the first infall of a halo occurs.
def find_first_infall(halo_df, host_rvir, vmax_min, vmax_max):
    num_elem = len(halo_df)
    for i in range(num_elem):
        # Find infalling halos
        if (np.flip(halo_df.dist.values)[i] < np.flip(host_rvir)[i]):
            infall_halo = halo_df.iloc[num_elem - (i+1)]
            # Check if vmax_in < Vmax < vmax_max
            if (vmax_min <= infall_halo.vmax <= vmax_max):
                # Return infall scale factor (scale factor of snapshot that comes immediately before ith snapshot)
                return(np.flip(halo_df.scale.values)[i-1])
            # The script ends after finding the first infall.
            return       
        
def get_density(r_array, dataframe):
    # The first shell is from r=0 to r=r_array[0]
    # All other shell has a width r_array[i]-r_array[i-1]
    # Making an array copy achieves the above without a loop.
    npts=len(r_array)-1
    dens_array=np.zeros(npts)
    v_circ_array=np.zeros(npts)
    num_enclosed=np.zeros(npts)
    
    inner_array=r_array[:-1]
    outer_array=r_array[1:]
    
    mid_array=(outer_array+inner_array)/2

    shell_vol=4.*np.pi/3. * (outer_array**3 - inner_array**3)
    
    # Each loop iteration computes, at is radius, various values:
    # shell mass and density, and enclosed mass and v_circ.
    
    # Distinguish between Pandas dataframe and array
    if isinstance(dataframe, pd.DataFrame):
        for i in range(len(outer_array)):
            p_inner=np.where(dataframe.dist.values <= inner_array[i])
            p_outer=np.where(dataframe.dist.values <= outer_array[i])
        
            num_inner=len(p_inner[0])
            num_outer=len(p_outer[0])
            num_shell=num_outer-num_inner
        
            shell_mass=num_shell * part_mass
            dens_array[i]=shell_mass/shell_vol[i]

            # Compute V_c(r)
            # Use outer_array - extends to R_vir
            p_inside=np.where(dataframe.dist.values <= outer_array[i])
            num_enclosed[i]=len(p_inside[0])
            mass_inside=part_mass*len(p_inside[0])
            v_circ_array[i]=np.sqrt(G*mass_inside / outer_array[i])
    
    else:
        # The input already is a distance array
        for i in range(len(outer_array)):
            p_inner=np.where(dataframe <= inner_array[i])
            p_outer=np.where(dataframe <= outer_array[i])
        
            num_inner=len(p_inner[0])
            num_outer=len(p_outer[0])
            num_shell=num_outer-num_inner
        
            shell_mass=num_shell * part_mass
            dens_array[i]=shell_mass/shell_vol[i]

            # Compute V_c(r)
            p_inside=np.where(dataframe <= outer_array[i])
            num_enclosed[i]=len(p_inside[0])
            mass_inside=part_mass*len(p_inside[0])
            v_circ_array[i]=np.sqrt(G*mass_inside / outer_array[i])
            
    # Return [mid_point array, dens_array, v_circ_array, num_enclosed]
    return(mid_array, outer_array, dens_array, v_circ_array, num_enclosed)
#
def compute_density_profile_sorted_pdist(distance_arr, particle_mass, first_idx, last_idx):
    '''
    * Function to compute the density profile, rho(r), for the given halo particle distribution, using logarithmically binned (by distance) shells.
    - distance_arr: a sorted particle distance array.
    - particle_mass: mass of the particle. It is assumed that all particles have the same mass, so it's a number rather than an array.
    '''
    '''
    if num_part < 100:
        last_idx = -int(np.rint(0.1 * num_part))
    else:
        last_idx = -10
    if num_part >= 1000:
        last_idx = -50
        print(distance_arr[last_idx:])
    else:
        last_idx = -int(np.rint(0.1 * num_part))
    '''
    # Set the number of bins to use.
    num_part = len(distance_arr)
    if num_part <= 100:
        # For very few particles, try to have around 10 particles, on average, in each bin. Particles are not evenly distributed, so bins will not have 10 particles each, but reducing the number of bins below 10 for very small number of particles is a good idea. 
        num_bins = num_part // 10
        #num_bins = 10
    elif num_part <= 500:
        # For between 100 and 500, use 10 bins.
        num_bins = 10
    elif num_part <= 1000:
        # For between 500 and 1000, try to have around 50 particles, on average, in each bin, up to 20 bins.
        num_bins = num_part // 50
    elif num_part <= 3000:
        # For between 1000 and 3000, use 20 bins.
        num_bins = 20
    elif num_part <= 7500:
        # For between 2000 and 7500, try to have around 150 particles, on average, in each bin, up to 50 bins.
        num_bins = num_part // 150
    else:
        # For above 7500, use 50 bins.
        num_bins = 50
    # Get the minimum and maximum distance bin values.
    r_min = distance_arr[first_idx]-0.001
    r_max = distance_arr[last_idx]+0.001
    r_min_log = np.log10(r_min)
    r_max_log = np.log10(r_max)
    # Set distance bins.
    log_r_bins = np.logspace(r_min_log, r_max_log, num_bins+1)
    bin_left = log_r_bins[:-1]
    bin_right = log_r_bins[1:]
    # sqrt(a * b) works because the midpoint of two points logarithmically is exp((log(a) + log(b))/2).
    bin_mid = (bin_left * bin_right) ** 0.5
    # Compute the volume of the bins.
    volume_shell = 4. * np.pi / 3. * (bin_right**3. - bin_left**3.)
    # Number of particles within the bin left points
    num_within_left = np.searchsorted(distance_arr, bin_left)
    # Number of particles within the bin right points
    num_within_right = np.searchsorted(distance_arr, bin_right)
    # Number of enclosed particles for distance points in bin_mid.
    num_enclosed = np.searchsorted(distance_arr, bin_mid)
    # Number of particles within shells
    num_shell = num_within_right - num_within_left
    # Mass within shells
    mass_shell = particle_mass * num_shell
    # Density within shells
    density_shell = mass_shell / volume_shell
    return(density_shell, bin_mid, num_enclosed)
#
def compute_density_profile(coords, halo_com, r_last, r_min_idx, num_bins, p_mass):
    '''
    * Function to compute the density profile of halo particles for one halo at one snapshot.
    - Returns density, enclosed particle number, bin-mid, and bin-right arrays.
    coords: an array with particle coordinates
    halo_com: COM of the halo
    r_last: Last radial bin value
    r_min_idx: first radial bin index - same index for all snapshots.
    num_bins: number of bins to use
    '''

    # Compute particles' distance using their coordinates and the COM.
    part_dist = np.linalg.norm(coords - halo_com, None, 1)

    # coords is not actually coordinates - it contains particles distances - need to fix it.
    #idx_to_use = np.where(part_dist <= 50)[0]
    #dist_to_use = part_dist[idx_to_use]
    
    sort_idx = np.argsort(part_dist)
    sorted_dist = part_dist[sort_idx]
    
    # r_min is not the smallest distance value:
    # The first few bins often have too few particles if the first bin is from the 
    r_min = np.log10(sorted_dist[r_min_idx])
    #r_min = np.log(r_first)
    r_max = np.log10(r_last)
    #r_max = np.log10(sorted_dist[-1])
    
    # Set radial bins.
    log_r = np.logspace(r_min, r_max, num_bins+1)
    inner_r = log_r[:-1]
    outer_r = log_r[1:]
    
    # Midpoint of the bin
    mid_r = (inner_r * outer_r)**0.5
    
    # Width of the bin
    shell_width = np.log(outer_r) - np.log(inner_r)
    
    # Compute density and number of enclosed particles.
    inner_locs = np.searchsorted(sorted_dist, inner_r)
    outer_locs = np.searchsorted(sorted_dist, outer_r)
    num_shell = outer_locs - inner_locs
    shell_mass = p_mass * num_shell
    dens_arr = shell_mass / shell_width / (4*np.pi*mid_r**3)
    num_enclosed = np.searchsorted(sorted_dist, mid_r)
    return(dens_arr, num_enclosed, inner_r, mid_r, outer_r, shell_width)
#
def cm(fname, nofile=False, centered=False, pmass=[0e0], r_stop=None, num_part=100, nel_lim=75, print_statement=True, **args):
    """compute center of mass of a GADGET snapshot.  

    INPUTS:
    fname  -- name of GADGET snapshot file, or, if nofile=True, an array of
    particle positions. 
    num_part -- the original value was 1000?
    nel_lim -- the original value was 500?
    OPTIONAL INPUTS:
    nofile=False  -- if True, then assume that 'fname' is actually an Nx3 array
    with positions. 
    centered=False  -- if True, then return Nx3 array in CM frame rather than
    the COM. 
    pmass=[0e0]  -- use a vector containing mass of each particle in computing
    the center of mass 

    **args:  any argument that can be passed to get_pos
    """
    # only do things iteratively for np >= nplim
    nplim=num_part

    if nofile == True:
        pos=fname.copy()
    else:
        pos=get_pos(fname, **args)

    if r_stop is None:
        r_stop=1e-10

    ntot=pos.shape[0]
    # if length of pmass is >1, then presumably we are trying to specify the
    # particle masses explicitly:
    if len(pmass) != 1:
        if len(pmass) != ntot:
            print('if specified, pmass must be an array with '+
                  'the same length as the array of positions')
            return
        else:
            # need to reshape particle mass array so that it can be multiplied
   # by position array in a consistent manner
            pmass=pmass.reshape(-1,1)

    # get center of mass of array, as defined by (sum(x_i))/N_i
    if len(pmass) == 1: 
        cm0=pos.astype('float64').mean(axis=0).astype('float32')
    else: 
        cm0=(pos.astype('float64')*pmass).mean(axis=0).astype('float32')/pmass.mean()

    if ntot < nplim:
        if print_statement:
            print('N_p is below limit for iterative CM determination; ' + 
              'computing CM for all particles directly.')
            print('Center of mass is ', cm0)
        cmtemp2=cm0
    else:
        # get radii in new CM frame
        radi=((pos-cm0)**2).sum(axis=1)**0.5

        # get radius of sphere holding 85% of particles, and locations of
        # particles within this sphere:
        tempi=radi.argsort()[int(ntot*0.85)]
        rmax=radi[tempi]
        locs=np.ravel((radi < rmax).nonzero())
        nel=np.size(locs)
        #print('number of elements within ' , rmax , ' kpc is ', nel)
        # compute CM of this subset of particles
        if len(pmass) == 1:
            cmn=(pos[locs,:]).mean(axis=0).astype('float32')
        else:
            cmn=(pos[locs,:].astype('float64')*pmass[locs]).mean(axis=0).astype('float32')/(pmass[locs]).mean()

        #print(cm0)
        #print(cmn)
        cmtemp1=cm0
        cmtemp2=cmn

        # once radius is below limitc, reduce search radius by 5% rather than 15%.
        rmax1=rmax/4.
        # iteratively determine COM:
        count=0
        # while nel >= 1000 and nel >= ntot/100:
        while nel >= nel_lim:
            if rmax >= rmax1:
                rmax=rmax*0.85
            else:
                rmax=rmax*0.95

            rad=((pos-cmtemp2)**2).sum(axis=1)**0.5
            locs=np.ravel((rad < rmax).nonzero())
            nel=np.size(locs)
            cmtemp1=cmtemp2
            if len(pmass) == 1:
                cmtemp2=pos[locs,:].astype('float64').mean(axis=0).astype('float32')
            else:
                cmtemp2=(pos[locs,:].astype('float64')*pmass[locs]).mean(axis=0).astype('float32')/pmass[locs].mean()
            count += 1
            if nel <= ntot/500:
                break
            if rmax < r_stop:
                break
        # del radi, tempi, locs, rad
        #print('number of iterations was ', count, nel)
        #print('Center of mass is ', cmtemp2, ' within ', rmax, ' kpc')
        #print('change from previous iteration was ', cmtemp2-cmtemp1)

    if centered == True:
        pos=pos-cmtemp2
        return pos
    else:
        return cmtemp2
    
# NFW profile
def nfw(r, rs, rho_0):
    rho=rho_0/((r/rs)*(1+r/rs)**2)
    return(rho)

def concentration(vcirc_arr, dist_arr):
    vmax_list = []
    rmax_list = []
    cv_list = []
    
    v_half_max_list = []
    r_half_max_list = []
    cv_half_list = []
    
    for i in range(len(vcirc_arr)):
        # Find Vmax and Rmax
        v_arr = vcirc_arr[i]
        d_arr = dist_arr[i]
        
        vmax_idx = v_arr.argmax()
        vmax = v_arr[vmax_idx]
        rmax = d_arr[vmax_idx]
        
        # Find V_max/2 and R_vmax/2
        v_max_over_2 = vmax/2
        vcirc_b4_vmax = v_arr[:vmax_idx]
        
        v_half_idx = (np.abs(vcirc_b4_vmax - v_max_over_2)).argmin()
        v_half_max = vcirc_b4_vmax[v_half_idx]
        r_half_max = d_arr[v_half_idx]
        
        # Compute C_v
        cv = 2*(vmax/(H0*rmax))**2
        
        # Compute C_v/2
        cv_half = 0.5* (vmax/(H0 * r_half_max))**2
        
        # Append to output lists
        vmax_list.append(vmax)
        rmax_list.append(rmax)
        v_half_max_list.append(v_half_max)
        r_half_max_list.append(r_half_max)
        cv_list.append(cv)
        cv_half_list.append(cv_half)
        
    return(cv_list, cv_half_list, vmax_list, rmax_list, v_half_max_list, r_half_max_list)
 
    
def concentration_one_halo(vcirc_arr, dist_arr):
    # Find Vmax and Rmax
    vmax_idx = vcirc_arr.argmax()
    vmax = vcirc_arr[vmax_idx]
    rmax = dist_arr[vmax_idx]
    
    '''
    # Find V_max/2 and R_vmax/2
    v_max_over_2 = vmax/2
    vcirc_b4_vmax = v_arr[:vmax_idx]
    
    v_half_idx = (np.abs(vcirc_b4_vmax - v_max_over_2)).argmin()
    v_half_max = vcirc_b4_vmax[v_half_idx]
    r_half_max = d_arr[v_half_idx]
    '''
        
    # Compute C_v
    cv = 2.*(vmax/(H0*rmax))**2.
        
    # Compute C_v/2
    # cv_half = 0.5* (vmax/(H0 * r_half_max))**2
    
    # Return concentration, Vmax, Rmax
    return(cv, vmax, rmax)


def compute_cv_halos(halo_list, run_type, snap_char):
    '''
    *** Use function vmax_all_snaps instead to compute c_v for all snapshots.
    * Function to compute C_v for halos in the halo object list.
    - halo_list: list of halo objects.
    - run_type: DMO or Disk
    - snap_char: DMO, Disk, or 
    '''
    cv_list = []
    vmax_list = []
    rmax_list = []
    for i in range(len(halo_list)):
        current_halo = halo_list[i]
        
        if run_type == 'DMO':
            if snap_char == 'infall':
                vcirc = current_halo.dmo_vcirc[0]
                pdist = current_halo.dmo_pdist[0]
                
            elif snap_char == 'z0':
                vcirc = current_halo.dmo_vcirc[-1]
                pdist = current_halo.dmo_pdist[-1]
                
        elif run_type == 'Disk':
            if snap_char == 'infall':
                vcirc = current_halo.disk_vcirc[0]
                pdist = current_halo.disk_pdist[0]
                
            elif snap_char == 'z0':
                vcirc = current_halo.disk_vcirc[-1]
                pdist = current_halo.disk_pdist[-1]
                
        cv, vmax, rmax = concentration_one_halo(vcirc, pdist)
        
        # Append to lists.
        cv_list.append(cv)
        vmax_list.append(vmax)
        rmax_list.append(rmax)
        
    return(cv_list, vmax_list, rmax_list)
            
def get_inner_most_particles(num_part, hID, out_dir, sim_type):
    
    # Get halo index
    halo_idx = np.where(hID_dmo_arr == hID)[0][0]
    
    if sim_type == 'dmo':
        # Do DMO stuff
        return()
        
        
    elif sim_type == 'disk':
        # Do Disk stuff
        scale_infall = np.array(infall_scale_list)[halo_idx]
            
        # Halos have the same infall snapshot number for DMO and Disk
        snap = np.array(infall_snap_list_dmo)[halo_idx]
        rvir_infall = np.array(rvir_list_infall_disk)[halo_idx] / scale_infall  # ckpc
        com_infall = np.array(com_list_infall_disk)[halo_idx] / scale_infall   #ckpc
            
        # Halo particle coordinates and IDs
        # If infall snap says 152 for the disk run, it means its infall was before snapshot 38 and
        # its infall snapshot should be read in from the DMO run.
        if snap < 38:
            print(f'* Infall snapshot is {snap}. DMO particles will be used for infall.')
            infall_coord = infall_coord_list_dmo[halo_idx]
            infall_pID = infall_pID_list_dmo[halo_idx]
                
        else:
            infall_coord = infall_coord_list_disk[halo_idx]
            infall_pID = infall_pID_list_disk[halo_idx]
        
    # Compute particle distance
    part_dist = np.linalg.norm(infall_coord - com_infall, None, 1)
    argsort_idx = np.argsort(part_dist)
    sorted_part_dist = part_dist[argsort_idx]
    sorted_coords = infall_coord[argsort_idx]
    
    # sorted_pID is NOT sorted by particle IDs
    # - these are particle IDs corresponding to sorted particle distances.
    sorted_pID = infall_pID[argsort_idx]
    
    # Take the inner most particles specified by num_part.
    inner_most_pID = sorted_pID[:num_part]
    inner_most_coords = sorted_coords[:num_part]

    # Save to .hdf5 file
    out_name = f'{out_dir}inner_most_particles/hID_{hID}_snap_{snap}.hdf5'
    dset_name_pID = 'Particle_IDs'
    dset_name_coords = 'Coordinates'
    with h5py.File(out_name, 'a') as out_hf:
        # One group per snapshot
        grp = out_hf.require_group(f'snapshot_{snap}')
        
        # Two datasets per group (snapshot)
        if dset_name_pID not in grp:
            grp.create_dataset(dset_name_pID, data = inner_most_pID)
            print(f'* dataset /snapshot_{snap}/{dset_name_pID} is created.')
            
        else:
            print(f'* dataset /snapshot_{snap}/{dset_name_pID} already exists and is skipped.')
            
        if dset_name_coords not in grp:
            grp.create_dataset(dset_name_coords, data = inner_most_coords)
            print(f'* dataset /snapshot_{snap}/{dset_name_coords} is created.')
            
        else:
            print(f'* dataset /snapshot_{snap}/{dset_name_coords} already exists and is skipped.')
    print(out_name)
    
    # Return result
    return(inner_most_coords)        
  

    

def get_halo_particle_file_names(dat_dir):
    '''
    * Function to obtain halo particle file names.
    - Returns 3 lists containing halo IDs, infall snapshot numbers, and full file names, sorted by halo IDs.
    '''
    #
    # Get all file names in the directory.
    _, _, fnames = next(os.walk(dat_dir))
    #
    # Remove '.DS_Store' from the filename list if it exists.
    if '.DS_Store' in fnames:
        fnames.remove('.DS_Store')
    #
    hID_list = []
    infall_snap_list = []
    full_name_list = []
    for fname in fnames:
        # Get the full file name.
        full_name = pathlib.PurePath(dat_dir, fname)
        #
        # Get halo ID and infall snapshot number from file names.
        hID = int(fname.split('_')[2])
        infall_snap = int(fname.split('_')[4].split('.')[0])
        #
        # Append to lists.
        hID_list.append(hID)
        infall_snap_list.append(infall_snap)
        full_name_list.append(full_name)
    #
    # Sort halo IDs.
    argsort_idx = np.argsort(hID_list)
    hID_arr = np.array(hID_list)[argsort_idx]
    infall_snap_arr = np.array(infall_snap_list)[argsort_idx]
    full_name_arr = np.array(full_name_list)[argsort_idx]
    #
    return(hID_arr, infall_snap_arr, full_name_arr)


def find_halo_in_rockstar(halo_obj):
    '''
    * Function to find halo entries from Rockstar catalog that match with the tracked halos.
    - All halos at infall should be identified in Rockstar.
    - Many tracked halos at z=0 will not be correctly identified - these halos need to be looked at later.
    - Returns a list containing:
    -- [0] Halo ID
    -- [1] List of DMO snapshot numbers with exactly one halo matched.
    -- [2] List of DMO snapshot numbers with no halo matched.
    -- [3] List of DMO snapshot numbers with more than one halo matched.
    -- [4] List of Disk snapshot numbers with exactly one halo matched.
    -- [5] List of Disk snapshot numbers with no halo matched.
    -- [6] List of Disk snapshot numbers with more than one halo matched. 
    '''
    hID = halo_obj.halo_ID
    dmo_halo_list = []
    disk_halo_list = []
    
    # Output list to return.
    output_list = []
    dmo_first_list = [] # DMO: contains snapshot numbers with exactly one halo matched.
    dmo_second_list = [] # DMO: contains snapshot numbers with no halo matched.
    dmo_third_list = [] # DMO: contains snapshot numbers with more than one halo matched.
    disk_first_list = [] # disk: contains snapshot numbers with exactly one halo matched.
    disk_second_list = [] # disk: contains snapshot numbers with no halo matched.
    disk_third_list = [] # disk: contains snapshot numbers with more than one halo matched.
    
    output_list.append(hID)
    
    for i in range(len(halo_obj.snapnums)):
        snapnum = halo_obj.snapnums[i]
    
        # Make catalog names.
        cat_name_dmo = f'{catalog_path}493_dmo_370pc/493_dmo_catalog_snap_{snapnum}.csv'
    
        if snapnum < 38:
            # If before snapshot 38, there is no Disk catalog.
            cat_name_disk = cat_name_dmo = f'{catalog_path}493_dmo_370pc/493_dmo_catalog_snap_{snapnum}.csv'
        
        else:
            cat_name_disk = f'{catalog_path}493_disk_370pc/493_disk_catalog_snap_{snapnum}.csv'
        
        # Open catalogs
        cat_dmo = pd.read_csv(cat_name_dmo)
        cat_disk = pd.read_csv(cat_name_disk)
    
        # Take halos within 1000 kpc from the host
        #within_1000_dmo = infall_cat_dmo[infall_cat_dmo['dist'] <= 1000]
        #within_1000_disk = infall_cat_disk[infall_cat_disk['dist'] <= 1000]
    
        # Find halo
        # DMO
        if i == 0:
            # Infall snapshot: Find DMO halos by the halo ID.
            cat_halo_dmo = cat_dmo.query('orig_id == @hID')
                    
        else:
            # Search for the halo using COM coordinates.
            pm_range = 0.003
            x_lower = halo_obj.dmo_com[i][0] / 1000 - pm_range
            x_upper = halo_obj.dmo_com[i][0] / 1000 + pm_range
            y_lower = halo_obj.dmo_com[i][1] / 1000 - pm_range
            y_upper = halo_obj.dmo_com[i][1] / 1000 + pm_range
            z_lower = halo_obj.dmo_com[i][2] / 1000 - pm_range
            z_upper = halo_obj.dmo_com[i][2] / 1000 + pm_range
            
            # Query the catalog for halos in the vicinity of the halo COM.
            cat_halo_dmo = cat_dmo.query('mvir < 5e11 & @x_lower < x < @x_upper & @y_lower < y < @y_upper & @z_lower < z < @z_upper')
            
        if len(cat_halo_dmo) == 0:
            # No halo is matched.
            dmo_halo_list.append(None)
            dmo_second_list.append(snapnum)
            #print(f"* Halo {hID} DMO at snapshot {snapnum}: No halo is matched.")
    
        else:
            # Halos are found.
            dmo_halo_list.append(cat_halo_dmo)
            if len(cat_halo_dmo) != 1:
                # Append snapshot number to dmo_third_list.
                dmo_third_list.append(snapnum)
                # Print a message if there are more than one halo found.
                #print(f"* Halo {hID} DMO at snapshot {snapnum}: More than 1 halo found!")
                
            else:
                # Exactly one halo is found.
                # Append snapshot number to dmo_first_list.
                dmo_first_list.append(snapnum)
                
        # Disk
        if i == 0:
            # Infall snapshot
            if snapnum < 38:
                # Using halo ID if infall snapshot < 38
                # Neeed to search for the halo otherwise
                cat_halo_disk = cat_dmo.query('orig_id == @hID')
                
            else:
                pm_range = 0.003
                x_lower = halo_obj.disk_com[i][0] / 1000 - pm_range       # COM is in ckpc.
                x_upper = halo_obj.disk_com[i][0] / 1000 + pm_range
                y_lower = halo_obj.disk_com[i][1] / 1000 - pm_range
                y_upper = halo_obj.disk_com[i][1] / 1000 + pm_range
                z_lower = halo_obj.disk_com[i][2] / 1000 - pm_range
                z_upper = halo_obj.disk_com[i][2] / 1000 + pm_range
                
                # Query the catalog for halos in the vicinity of the halo COM.
                cat_halo_disk = cat_disk.query('mvir < 5e11 & @x_lower < x < @x_upper & @y_lower < y < @y_upper & @z_lower < z < @z_upper')
                
        else:
            # Search for the halo using COM coordinates.
            pm_range = 0.004
            x_lower = halo_obj.disk_com[i][0] / 1000 - pm_range       # COM is in ckpc.
            x_upper = halo_obj.disk_com[i][0] / 1000 + pm_range
            y_lower = halo_obj.disk_com[i][1] / 1000 - pm_range
            y_upper = halo_obj.disk_com[i][1] / 1000 + pm_range
            z_lower = halo_obj.disk_com[i][2] / 1000 - pm_range
            z_upper = halo_obj.disk_com[i][2] / 1000 + pm_range
        
            # Query the catalog for halos in the vicinity of the halo COM.
            cat_halo_disk = cat_disk.query('mvir < 5e11 & @x_lower < x < @x_upper & @y_lower < y < @y_upper & @z_lower < z < @z_upper')
            
        if len(cat_halo_disk) == 0:
            # No halo is matched.
            disk_halo_list.append(None)
            disk_second_list.append(snapnum)
            #print(f"* Halo {hID} Disk at snapshot {snapnum}: No halo is matched.")
                
        else:
            # Halos are found.
            disk_halo_list.append(cat_halo_disk)
            if len(cat_halo_disk) != 1:
                # Append snapshot number to disk_third_list.
                disk_third_list.append(snapnum)
                # Print a message if there are more than one halo found.
                #print(f"* Halo {hID} Disk at snapshot {snapnum}: More than 1 halo found!")
                
            else:
                # Exactly one halo is found.
                # Append snapshot number to disk_first_list.
                disk_first_list.append(snapnum) 
    
    # Append snapshot number lists to the output list.
    output_list.append(dmo_first_list)
    output_list.append(dmo_second_list)
    output_list.append(dmo_third_list)
    output_list.append(disk_first_list)
    output_list.append(disk_second_list)
    output_list.append(disk_third_list)
    
    return(dmo_halo_list, disk_halo_list, output_list)

def not_found_in_rockstar(halo_obj_list):
    '''
    * Function to find halo entries that do not have matching entries in Rockstar catalogs.
    - Find all snapshots for each halo where a corresponding Rockstar entry cannot be found.
    matched_idx_by_snaps_dmo(disk): [[idx of halos matched for infall snap],[second],[third]]
    not_matched_idx_by_snaps_dmo(disk): [[idx of halos not matched for infall snap],[second],[third]]
    '''
    matched_idx_by_snaps_dmo = []
    matched_idx_by_snaps_disk = []
    not_matched_idx_by_snaps_dmo = []
    not_matched_idx_by_snaps_disk = []
    for i in range(len(halo_obj_list[0].snapnums)):
        matched_dmo = []
        matched_disk = []
        not_matched_idx_dmo = []
        not_matched_idx_disk = []
        for j in range(len(halo_obj_list)):
            current_entry = halo_obj_list[j]
            if current_entry.rockstar_halo_dmo[i] is None:
                # Append index.
                not_matched_idx_dmo.append(j)
                
            else:
                matched_dmo.append(j)
                
            if current_entry.rockstar_halo_disk[i] is None:
                not_matched_idx_disk.append(j)
            
            else:
                matched_disk.append(j)
                
        # Append to big lists.
        matched_idx_by_snaps_dmo.append(matched_dmo)
        matched_idx_by_snaps_disk.append(matched_disk)
        not_matched_idx_by_snaps_dmo.append(not_matched_idx_dmo)
        not_matched_idx_by_snaps_disk.append(not_matched_idx_disk)
        
    return(matched_idx_by_snaps_dmo, not_matched_idx_by_snaps_dmo, matched_idx_by_snaps_disk, not_matched_idx_by_snaps_disk)


def track_inner_particles(halo_obj, num_part, run_type, out_name='name'):
    '''
    * Function to obtain N inner-most particles of a halo at infall and track to z=0.
    - Result is saved as a .hdf5 file with structure that resembles a GIZMO output.
    
    halo_obj: halo object to get particles from
    num_part: number of particles to get
    run_type: type of simulation
    '''
    halo_ID = halo_obj.halo_ID

    # Particles IDs are the same for DMO and Disk halos.
    pIDs = halo_obj.dmo_pIDs
    
    # Get coordinates, COM, and velocities.
    if run_type == 'DMO':
        coords = halo_obj.dmo_coords
        com = np.array(halo_obj.dmo_com)
        vels = halo_obj.dmo_vels
        
    elif run_type == 'Disk':
        coords = halo_obj.disk_coords
        com = np.array(halo_obj.disk_com)
        vels = halo_obj.disk_vels
        
    # Compute particle's distance then sort it.
    # Note: "sorted" doesn't mean the list/array is sorted in itself
    #       - it's sorted by particle distance.
    part_dist_infall = np.linalg.norm(coords[0] - com [0], None, 1)
    argsort_idx = np.argsort(part_dist_infall)
    
    # Take N inner-most particles.
    inner_most_pIDs = pIDs[argsort_idx][:num_part]
    inner_most_coords = []
    
    # Save inner-most particles for each snapshot.
    for i in range(len(coords)):
        inner_most = coords[i][argsort_idx][:num_part]
        inner_most_coords.append(inner_most)
        
    return(inner_most_coords)    
    '''
    inner_most_coords_infall = coords[0][argsort_idx][:num_part]
    inner_most_coords_z0 = coords[2][argsort_idx][:num_part]
    inner_most_vels_z0 = vels[2][argsort_idx][:num_part]
    masses = np.full(num_part, part_mass)
    metallicities = np.ones(num_part)
    SF_time = np.ones(num_part)

    
    with h5py.File(out_name, 'a') as f:
        grp = f.require_group('PartType4')
        grp.create_dataset('Coordinates', data = inner_most_coords_z0)
        grp.create_dataset('Velocities', data = inner_most_vels_z0)
        grp.create_dataset('ParticleIDs', data = inner_most_pIDs)
        grp.create_dataset('Masses', data = masses)
        grp.create_dataset('Metallicities', data = metallicities)
        grp.create_dataset('StarFormationTime', data = SF_time)
        
    print(f'* Halo {halo_ID}: tracked inner-most particles saved at')
    print(f'  {out_name}')
    '''
    
def update_halo_com(h_obj, snaps_to_update, num_part, run_type):
    '''
    *** Function track_inner_particles() returns inner-most particles for all snapshots for the given halo object.
        This function (update_halo_com) obtains inner-most particles for specified snapshots and computes COM using those particles.
        So it is a better function that track_inner_particles().
    - This function requires the snapshot numbers for which the COM needs to be updated.
    '''
    halo_snaps = h_obj.snapnums
    pIDs = h_obj.dmo_pIDs
    snap_idx = np.searchsorted(halo_snaps, snaps_to_update)
    
    # DMO
    if run_type == "DMO":
        # Get array indices corresponding to the snapshots of interest.
        coords_infall = h_obj.dmo_coords[0]
        coords_to_use = h_obj.dmo_coords[snap_idx]
        infall_com = h_obj.dmo_com[0]
        
    # Disk
    elif run_type == "Disk":
        # Get array indices corresponding to the snapshots of interest.
        coords_infall = h_obj.disk_coords[0]
        coords_to_use = h_obj.disk_coords[snap_idx]
        infall_com = h_obj.disk_com[0]
        
    # Compute particle's distance at infall then sort it.
    part_dist_infall = np.linalg.norm(coords_infall-infall_com, None, 1)
    argsort_idx = np.argsort(part_dist_infall)
    
    # Take N inner-most particles.
    inner_most_pIDs = pIDs[argsort_idx][:num_part]
    
    # Save inner-most particles for each snapshot.
    inner_most_coords = []
    new_coms = []
    for i in range(len(coords_to_use)):
        inner_most = coords_to_use[i][argsort_idx[:num_part]]
        
        # Compute COM using 1000 inner-most particles.
        cm_1000 = cm(inner_most, nofile=True, part_lim=100, nel_lim=100)
        
        # Append to lists.
        new_coms.append(cm_1000)
        inner_most_coords.append(inner_most)
        
    # Update COMs in the halo object.
    # DMO
    if run_type == 'DMO':
        for i in range(len(snap_idx)):
            idx = snap_idx[i]
            h_obj.dmo_com[idx] = new_coms[i]
    
    # Disk
    elif run_type == 'Disk':
        for i in range(len(snap_idx)):
            idx = snap_idx[i]
            h_obj.disk_com[idx] = new_coms[i]

def get_host_properties(snapnums, rockstar_cat_path='/Users/hk9457/Desktop/Hyunsu/Research/pELVIS/halo_493/halo/catalog/'):
    '''
    * Function to obtain host halo's properties at all snapshots.
    *** All distance values are in comoving units.
    - Takes in a list of snapshot numbers and the path to the parent directory for DMO and Disk catalog directories.
    - Returns two lists (DMO, Disk)containing:
    [0] Snapshot numbers
    [1] host centers 
    [2] host Rvirs
    for all snapshots.
    '''
    dmo_out_list = []
    disk_out_list = []
    
    dmo_out_list.append(snapnums)
    disk_out_list.append(snapnums)
    
    dmo_host_ct_list = []
    dmo_host_rvir_list = []
    
    disk_host_ct_list = []
    disk_host_rvir_list = []
    
    for i in range(len(snapnums)):
        snap = snapnums[i]
        dmo_cat_fname = f'{rockstar_cat_path}493_dmo_370pc/493_dmo_catalog_snap_{snap}.csv'
        
        # Disk catalogs only exist for snapshots 38 and higher.
        if snap < 38:
            # DMO and Disk runs are the same and only the DMO catalogs need to be opened.
            dmo_cat = pd.read_csv(dmo_cat_fname)
            disk_cat = dmo_cat
            
            
        else:
            disk_cat_fname = f'{rockstar_cat_path}493_disk_370pc/493_disk_catalog_snap_{snap}.csv'
            # Open catalogs.
            dmo_cat = pd.read_csv(dmo_cat_fname)
            disk_cat = pd.read_csv(disk_cat_fname)
        
        
        # Find the row index of the host halo.
        # Halo catalogs are assumed to be correctly sorted and centered around the host.
        dmo_host_idx = dmo_cat.query('dist == 0').index.values[0]
        disk_host_idx = disk_cat.query('dist == 0').index.values[0]
        
        # Get host center and rvir and append to lists.
        dmo_host_ct = dmo_cat.iloc[dmo_host_idx][['x','y','z']].values
        disk_host_ct = disk_cat.iloc[disk_host_idx][['x','y','z']].values
            
        dmo_host_rvir = dmo_cat.iloc[dmo_host_idx].rvir
        disk_host_rvir = disk_cat.iloc[disk_host_idx].rvir
            
        dmo_host_ct_list.append(dmo_host_ct)
        disk_host_ct_list.append(disk_host_ct)
            
        dmo_host_rvir_list.append(dmo_host_rvir)
        disk_host_rvir_list.append(disk_host_rvir)
        
    # Append results to the output lists - use list.extend([list1, list2]) to get [..., [list1], [list2]]
    dmo_out_list.extend([dmo_host_ct_list, dmo_host_rvir_list])
    disk_out_list.extend([disk_host_ct_list, disk_host_rvir_list])
    
    return(dmo_out_list, disk_out_list)
            
def half_mass_radius(part_coords, part_com):
    num_part=len(part_coords)
    num_part_half = int(num_part/2)
    m_total=num_part*part_mass
    m_half=num_part_half*part_mass
    part_dist=np.linalg.norm(part_coords - part_com, None, 1)
    
    # Sort particles by distance.
    sort_idx = np.argsort(part_dist)
    sorted_dist = part_dist[sort_idx]
    
    # Rmax, R_halfmax
    r_max = sorted_dist[-1]
    r_halfmass = sorted_dist[num_part_half]
    
    return(r_max, r_halfmass)
            

def vmax_all_snaps(halo_obj, run_type):
    '''
    * Function to compute Vmax, Rmax, and Cv for a halo at all snapshots.
    
    '''
    if run_type == 'DMO':
        vcirc_list = halo_obj.dmo_vcirc
        pdist_list = halo_obj.dmo_pdist    
    elif run_type == 'Disk':
        vcirc_list = halo_obj.disk_vcirc
        pdist_list = halo_obj.disk_pdist
        
    num_elem_use = int(0.4 * len(pdist_list[0]))
    
    vmax_list = []
    rmax_list = []

    for i in range(len(vcirc_list)):
        vcirc_use = vcirc_list[i][:num_elem_use]
        pdist_use = pdist_list[i][:num_elem_use]
        
        # Find Vmax.
        vmax_idx = np.argmax(vcirc_use)
        vmax = vcirc_use[vmax_idx]
        rmax = pdist_use[vmax_idx]

        
        vmax_list.append(vmax)
        rmax_list.append(rmax)
        
    cv_list = 2*(np.array(vmax_list) / (H0 * np.array(rmax_list)))**2
        
    return(vmax_list, rmax_list, cv_list)
#    
# This function is old: a more elaborate version is get_pericenters() in halo_analysis.py
def find_pericenters(dist_arr):
    '''
    * Function to obtain the number of pericentric passages for a subhalo's orbit.
    - dist_arr: input halo distance array
    - Returns a list with:
    -- [0] = list of pericenter indices
    -- [1] = list of pericenter values
    '''
    # Output list
    out_list = []
    for i in range(len(dist_arr)-4):
        current_set = dist_arr[i:i+5]
        min_idx = np.argmin(current_set)
        
        if min_idx == 2:
            peri_idx = i+2
            out_list.append(peri_idx)
        
    return(out_list) 

def match_halo_to_catalog_com(coord, cat_df, com_range):
    '''
    * A function to find a matching halo in my .csv halo catalogs made from Rockstar results.
    - One halo at a time.
    - There are other versions of it I wrote already, but I want to make it as generic as possible.
    - Input:
    
    - Output:
    -- A pandas dataframe entry for the matched halo.
    -- If there is no match, return None?
    '''
    
    x_use = coord[0]
    y_use = coord[1]
    z_use = coord[2]
    
    x_low = x_use - com_range
    x_upp = x_use + com_range
    y_low = y_use - com_range
    y_upp = y_use + com_range
    z_low = z_use - com_range
    z_upp = z_use + com_range
    
    q_by_coords = cat_df.query('@x_low < x < @x_upp & @y_low < y < @y_upp & @z_low < z < @z_upp')
    
    return(q_by_coords)

 
def find_main_branch_subtree(subtree_dataframe):
    '''
    * A function to find the main branch of a single first-level (from top) subtree found from the tree of the host.
    - For the cases I checked (~600), this always worked as the main branch spanned the same number of
          snapshots as the entire subtree.
    '''
    # Get the non-repeating scale factors from the subtree.
    tree_scales = np.unique(subtree_dataframe.scale.values)
    num_snaps = len(tree_scales)
    
    # Check if the subtree has just one tree: then it is the main tree.
    if len(subtree_dataframe) == num_snaps:
        return(subtree_dataframe)
    
    else:
        '''
        - There are repeating scale factors.
        - So there are multiple trees.
        - Find the main branch by taking the first N rows of the subtree where
          N = len(tree_scales).
        '''
        main_branch = subtree_dataframe[:num_snaps]
        return(main_branch)

def track_within_rmax(coords_infall, coords_track, infall_com, infall_rmax, rmax_multiple, infall_scale):
    '''
    * Coords_infall, coords_track, infall_com are in ckpc.
    * infall_rmax is in pkpc - needs to be converted to ckpc.
    * Particles at all snapshots are in the same order as the first snapshot (infall).
      - so I can use the same indices to track them.
      
    * coords_infall: particle coordinates at the infall snapshot.
    * coords_track: particle coordinates at snapshots to track.
    * rmax_multiple: a number to multiply the infall Rmax (infall_rmax) with - radius to track within.
    
    * Returns:
    - new_com_list: a list containing new COMs for all snapshots (infall and tracked).
    '''
    # An empty list to append result to.
    new_com_list = []
    
    # Append the infall COM to the result list.
    new_com_list.append(infall_com)
    
    # Compute particle's distance at infall then sort it.
    part_dist_infall = np.linalg.norm(coords_infall-infall_com, None, 1) # in ckpc.
    argsort_idx = np.argsort(part_dist_infall)
    sorted_infall_dist = part_dist_infall[argsort_idx]
    
    # Take all particles within Rmax at infall (or multiples of Rmax).
    rmax_track_within = infall_rmax * rmax_multiple / infall_scale # in ckpc
    # Use np.searchsorted on the sorted distance array.
    rmax_idx = np.searchsorted(sorted_infall_dist, rmax_track_within)
    
    # Taking the argsort indices up to Rmax.
    track_idx = argsort_idx[:rmax_idx]
    
    # Number of particles.
    num_part_tot = len(sorted_infall_dist)
    num_part_tracked = rmax_idx
    
    # Tracking halo particles for the rest of the snapshots.
    for i in range(len(coords_track)):
        inner_most_coords = coords_track[i][track_idx]
        
        # Compute COM.
        new_com = cm(inner_most_coords, nofile=True, part_lim=100, nel_lim=100)
        
        # Append to the final list.
        new_com_list.append(new_com)

    # Return results.    
    return(new_com_list, num_part_tot, num_part_tracked)

def get_particles_within(coords, com, r_within):
    '''
    * Function to select halo particles that are within a given R value from COM for one snapshot.
    * Input:
    - coords: array containing 3D coordinates for the halo particles.
    - com: center of mass.
    - r_within: the radius to select particles within.
    
    *** coords, com, and r_within all must be in the same units.
    '''
    
    # Particles' distance from COM.
    part_dist = np.linalg.norm(coords - com, None, 1)
    
    # Sort the distance array.
    argsort_idx = np.argsort(part_dist)
    sorted_dist = part_dist[argsort_idx]
    
    # Find particles within r_within.
    r_within_idx = np.searchsorted(sorted_dist, r_within)
    
    # Coordinates of selected particles.
    coords_within = coords[argsort_idx[:r_within_idx]]
    
    # Return result.
    return(coords_within)
    
def particle_PE_direct_sum(coords_compute, coords_all, part_mass):
    '''
    * Function to compute the potential energy of a given particle distribution.
    - For each ith particle, compute the gravitational potential energy contribution from all of the other particles by direct sum.
    - This function doesn't care whether the given coordinates are in pkpc or ckpc; it just computes the PE of the given distribution.
    - num_use: number of particles to compute the PE for - each particle still will have PE contribution coming from ALL other particles.
    - *** Potential energy will be computed for particles in coords_compute using the contributions from coords_all.

    - Note:
        - I tested using np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2) to compute the distances at once,
        - to see if avoiding doing the distance calculation N times reduces the computation time.
        - For small numbers of particles, there's a small amount of time reductions, BUT for large numbers of particles, it was actually much slower!
            - For ~10000 particles, the matrix version can be over 10 times slower.
        - I think it's because this creates a n x n matrix that can get very large.
        - I might want to implement a nearest neighbor search algorithm like scipy's kdtree.
    '''
    pot_e_arr = np.zeros(len(coords_compute))
    for i in range(len(coords_compute)):
        # Current particle: ith particle
        p_i = coords_compute[i]
        
        # Distance between ith particle and all other particles.
        dist_btw_parts = np.linalg.norm(coords_all - p_i, None, 1)
        # Remove the distance computation of itself.
        #dist_use = np.delete(dist_btw_parts, i)
        # Remove the distance computation of itself:
        # Explicitly look for a place where the distance is zero instead of just using the index i in case coords_all and coords_compute don't share the same indexing, for whatever reason. Using np.where() does not take long!
        itself_idx = np.where(dist_btw_parts == 0.)[0][0]
        dist_use = np.delete(dist_btw_parts, itself_idx)
        
        # Compute the potential energy.
        '''
        * phi_ij: gravitational energy contribution from jth particle to the ith particle.
          phi_i: gravitational energy of particle i: sum of phi_ij.
        '''
        phi_ij = -1. * G * part_mass * part_mass / dist_use
        phi_i = np.sum(phi_ij)
    
        # "Append" phi_i to the final array.
        pot_e_arr[i] = phi_i
        
    # Return the array containing the PE of each particle.
    return(pot_e_arr)


def bulk_velocity(vel_array):
    # number of elements.
    n = len(vel_array)
    # x, y, z
    x = vel_array[:,0]
    y = vel_array[:,1]
    z = vel_array[:,2]
    # Compute component-wise averages.
    x_ave = np.sum(x) / n
    y_ave = np.sum(y) / n
    z_ave = np.sum(z) / n
    # Bulk 3D velocity.
    v_bulk = np.array([x_ave, y_ave, z_ave])
    # Return the result.
    return(v_bulk)

# Halo class
class halo:
    '''
    !!!!!!!!!!! This one is outdated as of 06/13/2024 !!!!!!!!!!!!
    Use the halo class in halo_analysis.py
    * A class for handling halo particles and halo's properties.
    
    - coords, COM: in comoving units (ckpc)
    '''
    def __init__(self, hID):  # , hID, infall_snap
        '''
        * Method to assign instance attributes.
        '''
        self.halo_ID = hID
        #self.dir_path = dir_path
        #self.z_range = z_range
        #self.infall_snapnum = infall_snapnum
             
        
    def set_dmo(self, dmo_path, a_arr, z_arr):
        '''
        * Halo class method to set DMO particles as attributes to the halo class object.
        snapnums: Snapshot numbers for which the halo particles were tracked to.
                  DMO and Disk have the same snapshot numbers.
        dmo_pIDs: Halo particle IDs. Only one set because particle IDs are the same and in the same order
                  for all snapshots.
        dmo_coords (ckpc): Halo particle coordinates at each snapshot - saved as a list where each element is the
                coordinates at each snapshot.
        dmo_vels (km/s): Halo particle velocities at each snapshot - same format as coords.
        
        '''
        self.dmo_file_path = dmo_path

        # Use function "open_halo_particles_file"(defined outside of the class) to read in halo particles.
        # - "open_halo_particles_file" needs the full file path.
        snaps, coords, vels, pIDs = open_halo_particles_file(self.dmo_file_path)
        
        
        # Set DMO coordinates and DMO velocities as attributes.
        self.dmo_coords = coords
        self.dmo_vels = vels
        self.dmo_pIDs = pIDs
        
        # Set snapshot numbers as attributes and obtain corresponding scale factors.
        # Full scale factor list from the snapshot header file is needed.
        # snaps is a sorted array
        self.snapnums = snaps
        scales_arr = a_arr[snaps - 1]
        redshift_arr = z_arr[snaps - 1]
        
        '''
        for i in range(len(snaps)):
            matching_scale = full_a_arr[snaps[i] - 1]
            scales_list.append(matching_scale)
            redshift_list
        '''
            
        self.scale_factors = scales_arr
        self.redshifts = redshift_arr
        
    
    def set_disk(self, disk_path):
        '''
        * Halo class method to set Disk particles as attributes to the halo class object.
        coords (ckpc): halo particle coordinates at each snapshot - saved as a list where each element is the
                coordinates at each snapshot.
        vels (km/s): halo particle velocities at each snapshot - same format as coords.
        '''
        self.disk_file_path = disk_path
        
        # Use function "open_halo_particles_file"(defined outside of the class) to read in halo particles.
        # - "open_halo_particles_file" needs the full file path.
        snaps, coords, vels, pIDs = open_halo_particles_file(self.disk_file_path)
        
        # Set Disk coordinates and Disk velocities as attributes.
        self.disk_coords = coords
        self.disk_vels = vels
        #self.disk_pIDs = pIDs
      
    
    def compute_com(self):
        '''
        * Function to compute the center of mass of the particles, in ckpc.
        '''
        com_dmo_list = []
        com_disk_list = []
        for i in range(len(self.snapnums)):
            # Compute COM for DMO and Disk halo particles for each snapshot.
            com_dmo = cm(self.dmo_coords[i], nofile = True, num_part=100, nel_lim=100)
            com_disk = cm(self.disk_coords[i], nofile = True, num_part=100, nel_lim=100)
            
            # Append to lists.
            com_dmo_list.append(com_dmo)
            com_disk_list.append(com_disk)
            
        # Set COM lists as attributes.
        self.dmo_com = com_dmo_list
        self.disk_com = com_disk_list
            
    
    def compute_vcirc(self, run_type):
        '''
        * Function to compute the circular velocity of the halo particles.
        '''
        if run_type == 'DMO':
            vcirc_dmo_list = []
            sorted_pdist_dmo_list = []
            for i in range(len(self.snapnums)):
                # Convert from ckpc to pkpc for each snapshot for DMO halos.
                current_scale = self.scale_factors[i]
                com_phys_dmo = self.dmo_com[i] * current_scale
                coords_phys_dmo = self.dmo_coords[i] * current_scale
                
                # Use the function vcirc_particle_single_halo(com, coords) to compute Vcirc.
                vcirc_arr, sorted_pdist_arr = vcirc_particle_single_halo(com_phys_dmo, coords_phys_dmo)
            
                # Append to lists.
                vcirc_dmo_list.append(vcirc_arr)
                sorted_pdist_dmo_list.append(sorted_pdist_arr)
                
            # Set Vcirc arrays attributes.
            self.dmo_vcirc = vcirc_dmo_list
            
            # Set sorted particle distance arrays (that match Vcirc arrays element by element) attributes.
            self.dmo_pdist = sorted_pdist_dmo_list       
            
        elif run_type == 'Disk':
            vcirc_disk_list = []
            sorted_pdist_disk_list = []
            for i in range(len(self.snapnums)):
                # Convert from ckpc to pkpc for each snapshot for Disk halos.
                current_scale = self.scale_factors[i]
                com_phys_disk = self.disk_com[i] * current_scale
                coords_phys_disk = self.disk_coords[i] * current_scale
                
                # Use the function vcirc_particle_single_halo(com, coords) to compute Vcirc.
                vcirc_arr, sorted_pdist_arr = vcirc_particle_single_halo(com_phys_disk, coords_phys_disk)
            
                # Append to lists.
                vcirc_disk_list.append(vcirc_arr)
                sorted_pdist_disk_list.append(sorted_pdist_arr)
            
            # Set Vcirc arrays attributes.
            self.disk_vcirc = vcirc_disk_list
            
            # Set sorted particle distance arrays (that match Vcirc arrays element by element) attributes.
            self.disk_pdist = sorted_pdist_disk_list
             
        elif run_type == 'both': # This needs to be modified with vcirc_particle_single_halo().
            vcirc_test_list = []
            vcirc_dmo_list = []
            vcirc_disk_list = []
            sorted_pdist_dmo_list = []
            sorted_pdist_disk_list = []
            for i in range(len(self.snapnums)):
                # Convert from ckpc to pkpc for each snapshot for DMO and Disk halos.
                current_scale = self.scale_factors[i]
                com_phys_dmo = self.dmo_com[i] * current_scale
                coords_phys_dmo = self.dmo_coords[i] * current_scale
            
                com_phys_disk = self.disk_com[i] * current_scale
                coords_phys_disk = self.disk_coords[i] * current_scale
                
                # Compute the distance, from COM, for each particle for DMO and Disk halos.
                # in Physical kpc.
                part_dist_dmo = np.linalg.norm(com_phys_dmo - coords_phys_dmo, None, 1)
                part_dist_disk = np.linalg.norm(com_phys_disk - coords_phys_disk, None, 1)
            
                # Sort the distance arrays.
                pdist_dmo_sorted = np.sort(part_dist_dmo)
                pdist_disk_sorted = np.sort(part_dist_disk)
            
                # Compute Vcirc
                # Create an array with particle numbers.
                part_num_arr = np.arange(1, len(pdist_dmo_sorted)+1)
                vcirc_arr_dmo = np.sqrt(G * part_mass * part_num_arr / pdist_dmo_sorted)
                vcirc_arr_disk = np.sqrt(G * part_mass * part_num_arr / pdist_disk_sorted)
            
                # Find Vmax and Rmax then compute Cv.
                '''
                def concentration_one_halo(vcirc_arr, dist_arr):
                # Find Vmax and Rmax
                vmax_idx = vcirc_arr.argmax()
                vmax = vcirc_arr[vmax_idx]
                rmax = dist_arr[vmax_idx]

                # Compute C_v
                cv = 2*(vmax/(H0*rmax))**2
        
                # Compute C_v/2
                # cv_half = 0.5* (vmax/(H0 * r_half_max))**2
    
                # Return concentration, Vmax, Rmax
                return(cv, vmax, rmax)
            
                '''
            
                # Append to lists.
                vcirc_dmo_list.append(vcirc_arr_dmo)
                vcirc_disk_list.append(vcirc_arr_disk)
                sorted_pdist_dmo_list.append(pdist_dmo_sorted)
                sorted_pdist_disk_list.append(pdist_disk_sorted)

            # Set Vcirc arrays attributes.
            self.dmo_vcirc = vcirc_dmo_list
            self.disk_vcirc = vcirc_disk_list
                 
            # Set sorted particle distance arrays (that match Vcirc arrays element by element) attributes.
            self.dmo_pdist = sorted_pdist_dmo_list
            self.disk_pdist = sorted_pdist_disk_list
  
    def halo_in_rockstar(self):
        '''
        * Halo class method to find the halo entry in Rockstar catalog for each snapshot.
        - Uses an external function, "find_halo_in_rockstar(halo)"
        - Sets rockstar halo entried for DMO and Disk runs and
        - returns a list containing:
        -- [0] Halo ID
        -- [1] List of snapshot numbers with exactly one halo matched.
        -- [2] List of snapshot numbers with no halo matched.
        -- [3] List of snapshot numbers with more than one halo matched.
        '''
        cat_halo_dmo, cat_halo_disk, output_list = find_halo_in_rockstar(self)
        self.rockstar_halo_dmo = cat_halo_dmo
        self.rockstar_halo_disk = cat_halo_disk
        
        return(output_list)
        
    def set_host_properties(self, rockstar_cat_path='/Users/hk9457/Desktop/Hyunsu/Research/pELVIS/halo_493/halo/catalog/'):
        '''
        * Halo class method to set host halo's properties.
        *** All distance values are in comoving units.
        - Halo center
        - Halo Rvir
        '''
        
        dmo_host_ct_list = []
        disk_host_ct_list = []
        
        dmo_host_rvir_list = []
        disk_host_rvir_list = []
        for i in range(len(self.snapnums)):
            dmo_cat_fname = f'{rockstar_cat_path}493_dmo_370pc/493_dmo_catalog_snap_{self.snapnums[i]}.csv'
            
            if self.snapnums[i] < 38:
                disk_cat_fname = dmo_cat_fname
                
            else:
                disk_cat_fname = f'{rockstar_cat_path}493_disk_370pc/493_disk_catalog_snap_{self.snapnums[i]}.csv'
            
            # Open catalogs.
            dmo_cat = pd.read_csv(dmo_cat_fname)
            disk_cat = pd.read_csv(disk_cat_fname)
            
            # Find the row index of the host halo.
            dmo_host_idx = dmo_cat.query('dist == 0').index.values[0]
            disk_host_idx = disk_cat.query('dist == 0').index.values[0]
            
            dmo_host_ct = dmo_cat.iloc[dmo_host_idx][['x','y','z']].values
            disk_host_ct = disk_cat.iloc[disk_host_idx][['x','y','z']].values
            
            dmo_host_rvir = dmo_cat.iloc[dmo_host_idx].rvir
            disk_host_rvir = disk_cat.iloc[disk_host_idx].rvir
            
            dmo_host_ct_list.append(dmo_host_ct)
            disk_host_ct_list.append(disk_host_ct)
            
            dmo_host_rvir_list.append(dmo_host_rvir)
            disk_host_rvir_list.append(disk_host_rvir)
            
        self.host_halo_ct_dmo = np.array(dmo_host_ct_list)
        self.host_halo_ct_disk = np.array(disk_host_ct_list)
        self.host_rvir_dmo = np.array(dmo_host_rvir_list)
        self.host_rvir_disk = np.array(disk_host_rvir_list)
         
    def compute_density(self):
        '''
        * Halo class method to compute the density profile of the halo object for each snapshot.
        - Sets density, mid-bin, enclosed particle number, and right-bin as class attributes.
        - Uses function compute_density_profile(halo_obj, num_bins, p_mass, r_max)
        num_bins: number of bins to use.
        p_mass: mass of particles.
        r_last: outer most radius bin - set using Rvir at infall.
        
        '''

        # DMO and Disk
        # Halo particle coordinates - in comoving units and will be converted to physical units below.
        scale_factors = self.scale_factors
        coords_dmo = self.dmo_coords
        coords_disk = self.disk_coords
        
        coords_phys_dmo = coords_dmo.copy()
        coords_phys_disk = coords_disk.copy()
        
        # Halo COM - in comoving units and will be converted to physical units below.
        com_dmo = self.dmo_com
        com_disk = self.disk_com
        
        com_phys_dmo = com_dmo.copy()
        com_phys_disk = com_disk.copy()
        
        
        # Convert ckpc to pkpc.
        for i in range(len(scale_factors)):
            coords_phys_dmo[i] = coords_phys_dmo[i] * scale_factors[i]
            coords_phys_disk[i] = coords_phys_disk[i] * scale_factors[i]
            com_phys_dmo[i] = com_phys_dmo[i] * scale_factors[i]
            com_phys_disk[i] = com_phys_disk[i] * scale_factors[i]
        
        '''
        # Find r_min to use: Nth particle's distance for the DMO run at infall in physical units.
        part_dist_disk_infall = np.linalg.norm(coords_phys_disk[0] - com_phys_disk[0], None, 1)
        sort_idx = np.argsort(part_dist_disk_infall)
        sorted_dist = part_dist_disk_infall[sort_idx]
        
        r_first_disk = sorted_dist[40]
        '''
        
        
        # Last radial bin: 2*Rvir at infall for the DMO run in physical units.
        # r_last shouldn't be very different between DMO and Disk so just use the same value?
        r_last_dmo = 2.*self.rockstar_halo_dmo[0].rvir.values[0] * scale_factors[0]
        r_last_disk = r_last_dmo
        #r_last_disk = 2.*self.rockstar_halo_disk[0].rvir.values[0] * scale_factors[0]
        
        # Lists to append the results to.
        dmo_dens_list = []
        dmo_encl_list = []
        dmo_inner_r_list = []
        dmo_mid_r_list = []
        dmo_outer_r_list = []
        dmo_shell_width_list = []
        
        disk_dens_list = []
        disk_encl_list = []
        disk_inner_r_list = []
        disk_mid_r_list = []
        disk_outer_r_list = []
        disk_shell_width_list = []
        
        for i in range(len(self.snapnums)):
            # DMO
            dmo_dens, dmo_encl, dmo_inner_r, dmo_mid_r, dmo_outer_r, dmo_shell_width =\
            compute_density_profile(coords_phys_dmo[i], com_phys_dmo[i], r_last_dmo, r_min_idx=50, num_bins=25, p_mass=part_mass)
            
            # Disk
            disk_dens, disk_encl, disk_inner_r, disk_mid_r, disk_outer_r, disk_shell_width =\
            compute_density_profile(coords_phys_disk[i], com_phys_disk[i], r_last_disk, r_min_idx=25, num_bins=25, p_mass=part_mass)
            
            # Append to lists.
            dmo_dens_list.append(dmo_dens)
            dmo_encl_list.append(dmo_encl)
            dmo_inner_r_list.append(dmo_inner_r)
            dmo_mid_r_list.append(dmo_mid_r)
            dmo_outer_r_list.append(dmo_outer_r)
            dmo_shell_width_list.append(dmo_shell_width)
            
            disk_dens_list.append(disk_dens)
            disk_encl_list.append(disk_encl)
            disk_inner_r_list.append(disk_inner_r)
            disk_mid_r_list.append(disk_mid_r)
            disk_outer_r_list.append(disk_outer_r)
            disk_shell_width_list.append(disk_shell_width)
            
        # Assign results as class attributes.
        # Densities
        self.dmo_density = dmo_dens_list
        self.disk_density = disk_dens_list
        
        # DMO bins
        self.dmo_inner_r = dmo_inner_r_list
        self.dmo_mid_r = dmo_mid_r_list
        self.dmo_outer_r = dmo_outer_r_list
        self.dmo_shell_width = dmo_shell_width_list
        
        # Disk bins
        self.disk_inner_r = disk_inner_r_list
        self.disk_mid_r = disk_mid_r_list
        self.disk_outer_r = disk_outer_r_list
        self.disk_shell_width = disk_shell_width_list

        
