#####################################################################################
##### This script reads in the consistent-trees merger tree data (tree.hdf5)    #####
##### and makes a few useful, smaller data files.                               #####
##### author: Hyunsu Kong                                                       #####
##### email: kongh@rpi.edu                                                      #####
##### website: https://hyunsukong.github.io/                                    #####
##### GitHub (for Bloodhound):                                                  #####
#####################################################################################

'''
### This script will always be run within bloodhound.py.
# Input:
  - tree.hdf5
  - snapshot_redshift_scale-factor file

# Output:
  - Main branch data for the main host halo
  - Main branch data for all surviving halos
  - Main branch data for all destroyed subhalos of the main halo
  - Subhalo catalog containing all surviving and destroyed subhalos of the main halo

# Steps:
1. Read the tree.hdf5 data and convert it to a Pandas dataframe.


# Note:
  - Some parts of this script are rather inefficient: e.g., steps to add the scale factor array and distance array.
  - But these are relatively quick steps, so I am going to keep them inefficient.
  - I also could break the main function down a little, but I didn't know it was going to get so long.
'''
####################################################################################
# Import libraries.
####################################################################################
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
import numpy as np
import os
import pandas as pd
pd.set_option('display.max_rows', 700)
pd.options.mode.chained_assignment = None # This is probably bad...
import sys
import time
####################################################################################
# Import local modules/libraries/scripts.
####################################################################################
import utilities
import halo_analysis
####################################################################################
# Input parameters
####################################################################################
#from config import parameter_fname

####################################################################################
# Functions
####################################################################################
def host_tree_subhalos_one_snap_late_to_early(loop_idx, host_tree_df, unique_time_idx):
    if loop_idx == 0:
        print(f"* Error: Last snapshot!")
    '''
    # Get indices to slice the full host halo tree to obtain all elements at the snapshot corresponding to loop_idx.
    '''
    # index+1 because the ith index is for the main branch.
    first_idx = unique_time_idx[loop_idx]+1
    # last_idx itself points to the host halo at the next snapshot.
    last_idx = unique_time_idx[loop_idx+1]
    # Slice the host halo tree by the indices: all host tree elements at the current snapshot other than the main branch.
    host_tree_full_snap = host_tree_df[first_idx:last_idx]
    return(host_tree_full_snap, first_idx, last_idx)
#
def subhalo_catalog_entry_from_tree(current_tree_df, infall_a, infall_idx, is_infalling, host_scale_arr, host_coords_arr_phys, host_radius_arr_phys, host_velocity_arr, parameter_dict, out_f):
    # Data is ordered from early to late snapshots.
    result_dict = {}
    current_tree_infall_snap = current_tree_df.iloc[infall_idx]
    current_tree_last_snap = current_tree_df.iloc[-1]
    current_tree_infall_to_last = current_tree_df[infall_idx:]
    current_tid = int(current_tree_infall_snap["tree.tid"])
    #print(current_tree_infall_to_last[['scale.factor', 'tree.tid', 'catalog.index', 'dindex', 'final.index', 'progenitor.main.index', 'progenitor.main.last.dindex', 'progenitor.number', 'tid']])
    # Find Vpeak: largest vel.circ.max the subhalo has ever had, regardless of whether it was a subhalo at the time or not.
    vmax_arr = current_tree_df['vel.circ.max'].values
    vpeak_idx = np.argmax(vmax_arr)
    # Find Mpeak: largest mass.vir the subhalo has ever had, regardless of whether it was a subhalo at the time or not.
    mvir_arr = current_tree_df['mass.vir'].values
    mpeak_idx = np.argmax(mvir_arr)
    '''
    # Pericenter information
      - Interpolate subhalo and host coordinates to compute the distance with more refined time steps.
      - Interpolating coordinates instead of the distance array gives a more accurate result.
      - Use snapshots from N snapshots before the first infall:
        1) helps a lot with edge cases where the first pericenter is close to the first infall.
        2) without having to deal with early wiggles not related to the host halo.
        3) set N = "peri.refine.half.width" + "peri.vrad.buffer"+1 or half of infall_idx.
    '''
    #N_desired = parameter_dict["peri.refine.half.width"]+parameter_dict["peri.vrad.buffer"]+1
    N_desired = parameter_dict["peri.refine.half.width"]
    if infall_idx >= N_desired:
        N = N_desired
    else:
        N = infall_idx // 2
    start_idx = infall_idx - N
    current_tree_start_to_last = current_tree_df[start_idx:]
    current_tree_scale_start_to_last = current_tree_start_to_last['scale.factor'].values
    current_tree_coords_start_to_last = current_tree_start_to_last[['x','y', 'z']].values # non-h ckpc
    current_tree_coords_phys_start_to_last = halo_analysis.get_physical_x_y_z(current_tree_coords_start_to_last, current_tree_scale_start_to_last)
    # Find indices to slice the data for the host: idx_2 handles subtrees.
    #idx_1 = np.where(np.isclose(host_scale_arr, current_tree_scale_infall_to_last[0], atol=1e-04))[0][0]
    #idx_2 = np.where(np.isclose(host_scale_arr, current_tree_scale_infall_to_last[-1], atol=1e-04))[0][0]
    idx_1 = np.where(np.isclose(host_scale_arr, current_tree_scale_start_to_last[0], atol=1e-04))[0][0]
    idx_2 = np.where(np.isclose(host_scale_arr, current_tree_scale_start_to_last[-1], atol=1e-04))[0][0]
    host_coords_phys_start_to_last = host_coords_arr_phys[idx_1:idx_2+1]
    host_radius_phys_start_to_last = host_radius_arr_phys[idx_1:idx_2+1]
    host_velocity_start_to_last = host_velocity_arr[idx_1:idx_2+1]
    # Get the radial velocities of the subhalo.
    current_tree_radvels_start_to_last = halo_analysis.get_radial_velocities_tree(current_tree_start_to_last, current_tree_coords_phys_start_to_last, host_coords_phys_start_to_last, host_velocity_start_to_last)
    # Use the pericenter function to compute the pericenter information.
    pericenters, pericenter_scales, closest_peri_dist, closest_peri_scale = halo_analysis.compute_pericenters(current_tree_coords_phys_start_to_last, current_tree_radvels_start_to_last, host_coords_phys_start_to_last, current_tree_scale_start_to_last, parameter_dict, current_tid, out_f)
    #pericenters, pericenter_scales, closest_peri_dist, closest_peri_scale = halo_analysis.compute_pericenters(current_tree_coords_phys_start_to_last, current_tree_radvels_start_to_last, host_coords_phys_start_to_last, current_tree_scale_start_to_last, host_radius_phys_start_to_last, is_infalling, parameter_dict, current_tid, out_f) # Old version: took host_radius_phys_start_to_last and is_infalling as input, but they were never actually used. So I removed them.
    # Save subhalo properties.
    result_dict["ID.tree"] = int(current_tree_infall_snap['tree.tid'])
    result_dict["ID.halo.infall"] = int(current_tree_infall_snap['catalog.index']) # catalog.index (i.e. halo ID) at the infall snapshot.
    result_dict["vmax"] = current_tree_last_snap['vel.circ.max'] # Vmax at the last snapshot.
    result_dict["vmax.infall"] = current_tree_infall_snap['vel.circ.max']
    result_dict["vmax.peak"] = vmax_arr[vpeak_idx]
    result_dict["scale.radius"] = current_tree_last_snap['scale.radius'] # scale radius at the last snapshot, assuming non-h pkpc for now.
    result_dict["scale.radius.klypin"] = current_tree_last_snap['scale.radius.klypin']
    result_dict["rmax"] = 2.1626 * current_tree_last_snap['scale.radius.klypin'] # rs_klypin should have been computed by rmax / 2.1626, so use it to get rmax.
    result_dict["rmax.infall"] = 2.1626 * current_tree_infall_snap['scale.radius.klypin']
    result_dict["rmax.peak"] = 2.1626 * current_tree_df.iloc[vpeak_idx]['scale.radius.klypin'] # rmax when vpeak happens
    result_dict["mass.vir"] = current_tree_last_snap['mass.vir'] # Mvir at the last snapshot
    result_dict["mass.vir.infall"] = current_tree_infall_snap['mass.vir'] # Mvir at the infall snapshot
    result_dict["mass.vir.peak"] = mvir_arr[mpeak_idx]
    result_dict["scale.factor.infall"] = infall_a # scale factor at the infall snapshot.
    result_dict["scale.factor.disrupt"] = current_tree_last_snap['scale.factor']
    result_dict["scale.factor.vmax.peak"] = current_tree_df.iloc[vpeak_idx]['scale.factor']
    result_dict["scale.factor.mass.vir.peak"] = current_tree_df.iloc[mpeak_idx]['scale.factor']
    result_dict["scale.factor.closest.pericenter"] = closest_peri_scale
    result_dict["radius"] = current_tree_last_snap['radius'] # radius of the halo at the last snapshot, assuming non-h pkpc for now.
    result_dict["x"] = current_tree_last_snap['x'] # non-h ckpc, might need to check for a factor of h later.
    result_dict["y"] = current_tree_last_snap['y']
    result_dict["z"] = current_tree_last_snap['z']
    result_dict["host.x"] = current_tree_last_snap['host.x'] # host-centric coordinate assuming non-h pkpc
    result_dict["host.y"] = current_tree_last_snap['host.y']
    result_dict["host.z"] = current_tree_last_snap['host.z']
    result_dict["vx"] = current_tree_last_snap['vx']
    result_dict["vy"] = current_tree_last_snap['vy']
    result_dict["vz"] = current_tree_last_snap['vz']
    result_dict["host.vx"] = current_tree_last_snap['host.vx'] # velocities w.r.t the center of the primary host: I find it, in general, for some reason, these to be slightly (~5-15 km/s) different from [vx, vy, vz] - [host's vx, vy, vz].
    result_dict["host.vy"] = current_tree_last_snap['host.vy']
    result_dict["host.vz"] = current_tree_last_snap['host.vz']
    result_dict["host.vrad"] = current_tree_last_snap['host.velocity.rad'] # radial velocity w.r.t the center of the primiary host.
    result_dict["distance.from.host.ckpc"] = current_tree_last_snap['distance.from.host.ckpc'] # Distance from the host halo at the last snapshot.
    result_dict["is.infalling"] = is_infalling # Whether the halo has an infall to the host halo.
    result_dict["closest.pericenter"] = closest_peri_dist # non-h pkpc
    result_dict["number.of.pericenters"] = len(pericenters)
    #
    return(result_dict)
#
def peri_parameter_dict(target_t_space_gyr, target_savgol_window_gyr, vrad_sign_frac_threshold, scale_arr, BH_parameters, out_f):
    # Cosmological parameters from the Bloodhound parameter file
    omega_l = BH_parameters['omega_l']
    omega_m = BH_parameters['omega_m']
    h = BH_parameters['h']
    H0 = 0.1*h #km/s/kpc
    G = BH_parameters['G']
    cosmo = FlatLambdaCDM(H0=H0*1000., Om0=omega_m, Tcmb0=2.725)
    #
    tlb_arr = cosmo.lookback_time(1./scale_arr -1.).value # in Gyr
    tlb_spacing_arr = np.absolute(tlb_arr[1:] - tlb_arr[:-1])
    median_tlb_spacing = np.median(tlb_spacing_arr)
    interp_factor = int(median_tlb_spacing / target_t_space_gyr)
    if interp_factor < 2:
        interp_factor = 2
    savgol_window = int(target_savgol_window_gyr / target_t_space_gyr * interp_factor)
    # savgol_filter window must be a positive odd integer.
    if savgol_window % 2 == 0:
        savgol_window-=1
    peri_refine_half_width = int(savgol_window/2)
    # Number of snapshots required in the subhalo data to interpolate distances
    min_snap_for_interp = int(savgol_window/interp_factor) + 1
    # buffer size (in indices) for checking the sign change of the radial velocity around the pericenter: radial velocities within the buffer size indices will not be considered for the radial velocity sign consistency check.
    peri_vrad_buffer_size = interp_factor
    vrad_sign_frac_threshold = 0.9
    print(f"* Target t_spacing for distance interpolation: {target_t_space_gyr:.3f} Gyr", flush=True, file=out_f)
    print(f"* Target time window for savgol_filter: {target_savgol_window_gyr:.3f} Gyr", flush=True, file=out_f)
    print(f"* Median t_spacing: {median_tlb_spacing:.3f} Gyr", flush=True, file=out_f)
    print(f"* min(t_spacing): {np.min(tlb_spacing_arr):.3f} Gyr", flush=True, file=out_f)
    print(f"* max(t_spacing): {np.max(tlb_spacing_arr):.3f} Gyr", flush=True, file=out_f)
    print(f"* Interpolation factor: {interp_factor}", flush=True, file=out_f)
    print(f"* savgol_filter window length: {savgol_window}", flush=True, file=out_f)
    print(f"* Half window for radial velocity checks: {peri_refine_half_width}", flush=True, file=out_f)
    print(f"* Buffer window for radial velocity checks: +-{peri_vrad_buffer_size} indices around the pericenter candidate index", flush=True, file=out_f)
    print(f"* Consistency fraction required for infalling and receding radial velocities: {vrad_sign_frac_threshold}", flush=True, file=out_f)
    print(f"", flush=True, file=out_f)
    #
    parameter_dict = {
        "interp.factor":interp_factor,
        "min.snaps.for.interp": min_snap_for_interp,
        "savgol.window.size": savgol_window,
        "savgol.polyorder": 3,
        "peri.refine.half.width":peri_refine_half_width,
        "peri.vrad.buffer": peri_vrad_buffer_size,
        "vrad.sign.frac.threshold": vrad_sign_frac_threshold
    }
    return(parameter_dict)
#
def wrapper_subhalo_catalog_from_tree(host_main_branch_df, main_branch_df, parameter_dict, catalog_dict, subhalo_type, out_f):
    tid_arr = np.unique(main_branch_df['tree.tid'].to_numpy())
    print(f"* Number of {subhalo_type} halos in the tree data: {len(tid_arr)}", flush=True, file=out_f)
    tree_grpby_tid = main_branch_df.groupby(by="tree.tid")
    host_tid = host_main_branch_df.iloc[0]['tree.tid']
    host_scale_arr = host_main_branch_df['scale.factor'].to_numpy()
    # halo radius, for 'default' overdensity definition of R_200m [kpc physical - this is from Andrew Wetzel's code, there's no guarantee this dataset used it.]. Not R_vir! 
    host_radius_arr = host_main_branch_df['radius'].to_numpy() # Not sure what the unit is... going to assume physical kpc for now.
    #host_radius_arr_comoving = host_radius_arr / host_scale_arr
    host_scale_arr_flipped = host_scale_arr[::-1] # Early to late
    #host_radius_arr_phys_flipped = host_radius_arr[::-1]
    host_radius_arr_flipped = host_radius_arr[::-1]
    #host_radius_arr_comoving_flipped = host_radius_arr_comoving[::-1] # Use comoving: 'distance.from.host.ckpc' in surv_main_branch_df is in ckpc.
    host_coords_arr_comoving_flipped = host_main_branch_df[['x','y','z']].to_numpy()[::-1] # Early to late
    host_coords_arr_phys_flipped = halo_analysis.get_physical_x_y_z(host_coords_arr_comoving_flipped, host_scale_arr_flipped)
    host_velocity_arr_flipped = host_main_branch_df[['vx', 'vy', 'vz']].to_numpy()[::-1]
    # Remove host halo's tree ID from surv_tid_arr.
    tid_arr = tid_arr[tid_arr != host_tid]
    # Counter for the number of subhalos to be saved.
    sub_counter = 0
    for i in range(len(tid_arr)):
        #i=i+72
        current_tid = tid_arr[i]
        #print(current_tid)
        current_tree_df = tree_grpby_tid.get_group(current_tid)
        current_tree_scale_arr = current_tree_df['scale.factor'].to_numpy()
        if np.min(current_tree_scale_arr) < np.min(host_scale_arr):
            # Subhalo's tree starts earlier than the host main branch: very rare.
            # In this case, discard data from those earlier snapshots: I don't see them ever being important.
            tree_first_idx = np.where(np.isclose(current_tree_scale_arr, np.min(host_scale_arr), atol=1e-4))[0][0]
            # Assuming the tree data are ordered from late to early times:
            current_tree_df = current_tree_df[:tree_first_idx+1]
        # Find the first infall scale factor.
        infall_a, infall_idx, is_infalling = halo_analysis.find_first_infall_from_tree(current_tree_df, host_scale_arr_flipped, host_radius_arr_flipped)
        if infall_a == -1.:
            # Current halo is NOT a subhalo, so skip it!
            continue
        sub_counter += 1
        # Flip the halo tree to early to late.
        current_tree_df = current_tree_df.iloc[::-1]
        current_catalog_entry_dict = subhalo_catalog_entry_from_tree(current_tree_df, infall_a, infall_idx, is_infalling, host_scale_arr_flipped, host_coords_arr_phys_flipped, host_radius_arr_flipped, host_velocity_arr_flipped, parameter_dict, out_f)
        # 
        key_names = list(current_catalog_entry_dict.keys())
        if sub_counter == 1:
            # If it's the first subhalo analyzed, use its catalog entry dictionary to initialize catalog_dict's key-value pairs.
            for j in range(len(key_names)):
                key_name = key_names[j]
                catalog_dict[subhalo_type][key_name] = []
        # Append the catalog entries to the result dictionary.
        for j in range(len(key_names)):
            key_name = key_names[j]
            catalog_dict[subhalo_type][key_name].append(current_catalog_entry_dict[key_name])
    return(catalog_dict)
####################################################################################
# Main function
####################################################################################
def main(BH_parameters, out_f):
    # Full path to the full merger tree file: tree.hdf5
    tree_hdf5_fname = BH_parameters['tree_hdf5_fname']
    # Snapshot-redshift-scale factor information for the simulation.
    snap_z_a_df = pd.DataFrame.from_dict(BH_parameters['time_info_dict'])
    snap_z_a_flipped_df = snap_z_a_df[::-1] # Flip to order it late to early times.
    print(f"* tree_hdf5_fname: {tree_hdf5_fname}", flush=True, file=out_f)
    # Read in the tree file.
    with h5py.File(tree_hdf5_fname, 'r') as tree_hf:
        # Keys: merger tree property names
        first_level_keys = list(tree_hf.keys())
        # 'final.index' will be used to group the tree data halo-by-halo.
        final_idx_arr = np.unique(tree_hf['final.index'])
        print(f"* Number of halos in the tree data: {len(final_idx_arr)}", flush=True, file=out_f)
        print("* Property names in the merger tree:", flush=True, file=out_f)
        print(f"  {first_level_keys}", flush=True, file=out_f)
        print("", flush=True, file=out_f)
        # Convert the HDF5 data to a dictionary then to a pandas dataframe.
        t_s_step = time.time()
        print(f"* Converting the tree data to a dictionary then to a pandas dataframe... ", end="", flush=True, file=out_f)
        tree_dict = {}
        for dict_key in first_level_keys:
            dict_key = dict_key.lower()
            # Skip simulation information entries.
            if "cosmology" in dict_key or "info" in dict_key:
                continue
            dict_value = tree_hf[dict_key][()]
            #print(dict_key, ": ", dict_value[918826])
            # Store 3D entries separately.
            if dict_key == 'host.distance':
                tree_dict['host.x'] = dict_value[:,0]
                tree_dict['host.y'] = dict_value[:,1]
                tree_dict['host.z'] = dict_value[:,2]
            elif dict_key == 'host.velocity':
                tree_dict['host.vx'] = dict_value[:,0]
                tree_dict['host.vy'] = dict_value[:,1]
                tree_dict['host.vz'] = dict_value[:,2]
            elif dict_key == 'position':
                tree_dict['x'] = dict_value[:,0]
                tree_dict['y'] = dict_value[:,1]
                tree_dict['z'] = dict_value[:,2]
            elif dict_key == 'velocity':
                tree_dict['vx'] = dict_value[:,0]
                tree_dict['vy'] = dict_value[:,1]
                tree_dict['vz'] = dict_value[:,2]
            else:
                tree_dict[dict_key] = dict_value
        tree_df = pd.DataFrame.from_dict(tree_dict)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Group the tree data halo-by-halo using 'final.index'.
    tree_grouped_by_final_tid = tree_df.groupby(by="final.index")
    '''
    ### Getting the main branch for all surviving halos
    '''
    t_s_step = time.time()
    print("\n### Extracting the main branch for all surviving halos... ", flush=True, file=out_f)
    main_branch_list = []
    host_candidate_idxs = []
    for i in range(len(final_idx_arr)):
        # Get the tree data for the current halo.
        current_ftid = final_idx_arr[i]
        current_tree = tree_grouped_by_final_tid.get_group(current_ftid)
        # Get the main branch for the current halo.
        main_progenitor_last_didx = int(current_tree.iloc[0]['progenitor.main.last.dindex'])
        main_branch = current_tree.query('`progenitor.main.last.dindex`==@main_progenitor_last_didx')
        current_tid = int(main_branch['tid'].iat[0])
        # Check, just in case, that all elements in the main branch are actually main branch elements.
        if len(np.where(main_branch['am.progenitor.main'].values==0)[0]) > 0:
            print("*** Something wrong: non-main branch element present!", flush=True, file=out_f)
            print(f"*** tid: {current_tid}, final.index: {int(current_ftid)}", flush=True, file=out_f)
        # Add a tree ID array.
        tid_arr = np.full(len(main_branch), fill_value=current_tid)
        main_branch.insert(0,"tree.tid", tid_arr)
        # Check if the current halo is a host halo.
        coords = main_branch[['host.x', 'host.y', 'host.z', 'host.vx', 'host.vy', 'host.vz']].to_numpy()
        central = main_branch['central.index'].to_numpy()
        if not coords.any() and (central < 0).all():
            # np.any() checks whether any element is non-zero (true).
            # not coords.any() then will return true if everything is zero.
            # 'central.index' < 0 means there's no host halo for the current halo, so IT is the host.
            host_candidate_idxs.append(i)
        # Append to the final list of dataframes.
        main_branch_list.append(main_branch)
        if i in [999, 9999, 19999, 29999, 49999, 69999, 89999]:
            t_e_check = time.time()
            print(f"  * {i+1} main branches extracted: ", end="", flush=True, file=out_f)
            utilities.print_time_taken(t_s_step, t_e_check, "*", True, out_f)
    t_e_step = time.time()
    print(f"  * All {i+1} main branches extracted: ", end="", flush=True, file=out_f)
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Check if there's a scale factor column: if not, create one.
    possible_scale_col_names = ["a", "scale", "scale.factor", "scale_factor"]
    col_names = main_branch_list[0].columns
    matches = [col for col in possible_scale_col_names if col in col_names]
    if matches:
        # A scale factor column already exists: use its data, but change the name to 'scale.factor' if it's different.
        existing_col = matches[0]
        if existing_col != "scale.factor":
            for df in main_branch_list:
                df["scale.factor"] = df.pop(existing_col) # rename
    else:
        t_s_step = time.time()
        # No scale factor column exists: create a new one.
        for current_main_branch in main_branch_list:
            current_snapshot_arr = current_main_branch['snapshot'].values
            #snapshot_indices = np.where(np.isin(snap_z_a_flipped_df['snapshot_numbers'].values, current_snapshot_arr))[0]
            # For performance optimization, dict(zip(snapshot_arr, scale_arr)), then .map() would be better than np.isin, but I don't care right now.
            snapshot_indices = np.isin(snap_z_a_flipped_df['snapshot_numbers'].values, current_snapshot_arr)
            # Scale factor array corresponding to the snapshot array.
            scale_arr = snap_z_a_flipped_df['scale_factors'].values[snapshot_indices]
            current_main_branch['scale.factor'] = scale_arr
        print("* scale.factor column added: ", end="", flush=True, file=out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Get the main branch of the host halo.
    if len(host_candidate_idxs)>1:
        # More than one host halo found: this could be a problem, or could be because it's a pair simulation.
        # Either way, I need to write a script to handle this case ...
        print("\n*** More than one candidate for the main host halo found!", flush=True, file=out_f)
        print("*** I need to write something to handle these cases, but I haven't done it yet...\n", flush=True, file=out_f)
        raise ValueError("*** More than one candidate for the main host halo found!")
    elif len(host_candidate_idxs) == 0:
        # No host halo found: this definitely is a problem.
        print("\n*** ERROR: no main host halo found! ***\n", flush=True, file=out_f)
        raise ValueError("*** ERROR: no main host halo found! ***")
    elif len(host_candidate_idxs) == 1:
        # Exactly one host halo candidate found: I think my criteria (all of host.x, host.y, host.z, host.vx, host.vy, host.vz being zero AND central.index < 0) is good enough to trust this candidate to be the main host halo.
        host_main_branch = main_branch_list[host_candidate_idxs[0]]
        host_tree_tid = int(host_main_branch['tree.tid'].iat[0])
        host_scales = host_main_branch['scale.factor'].to_numpy()
        host_coords = host_main_branch[['x','y','z']].to_numpy()
        print(f"* Host halo identified: tree.tid-{host_tree_tid}", flush=True, file=out_f)
    #
    # Add distance.from.host.ckpc column.
    t_s_step = time.time()
    for i in range(len(main_branch_list)):
        current_tree = main_branch_list[i]
        current_tree_scales = current_tree['scale.factor'].to_numpy()
        # computing the distances using host.x, host.y, host.z, which are in physical kpc.
        current_tree_dist_arr = np.linalg.norm(current_tree[['host.x', 'host.y', 'host.z']].to_numpy(), None, 1) / current_tree_scales # ckpc
        '''
        # Below computes the distances by subtracting the coordinates of the host halo from the current halo. I have verified that the results are identical to just using ['host.x', 'host.y', 'host.z'], so I am using the simpler method.
        # Leaving it commented out in case ['host.x', 'host.y', 'host.z'] doesn't exist: it should always exist?
        # computing the distances by actually subtracting the coordinates:
        current_tree_coords = current_tree[['x','y','z']].to_numpy()
        # Find the host tree's index corresponding to the first snapshot of the current tree.
        first_snap_idx = np.where(np.isclose(host_scales, current_tree_scales[-1], atol=1e-4))[0]
        idx = 1
        while len(first_snap_idx) == 0:
            print(idx, -1-idx)
            # If the current tree begins before the first snapshot of the host tree, take the next snapshot, then the next and so on.
            first_snap_idx = np.where(np.isclose(host_scales, current_tree_scales[-1 - idx], atol=1e-4))[0]
            idx+=1
        first_snap_idx = first_snap_idx[0]
        # Empty array of length = len(current_tree) instead of the number of selected snapshots because I want the result array to be of the same length as the tree.
        current_tree_dist_arr_1 = np.full(shape=len(current_tree_scales), fill_value=-1.)
        # Compute the halo distance at all selected snapshots.
        current_tree_dist_arr_1[:first_snap_idx+1] = np.linalg.norm(host_coords[:first_snap_idx+1] - current_tree_coords[:first_snap_idx+1], None, 1)
        '''
        # Add the distance array to the current dataframe.
        current_tree['distance.from.host.ckpc'] = current_tree_dist_arr
    print("* distance.from.host.ckpc column added: ", end="", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Replace the host halo data with the new one with the 'distance.from.host.ckpc' column: this column should contain zeros.
    host_main_branch = main_branch_list[host_candidate_idxs[0]]
    # Convert the list of main branch dataframes to one big dataframe.
    t_s_step = time.time()
    surv_main_branch_df = pd.concat(main_branch_list)
    print(f"### List of {len(main_branch_list)} main branch dataframes concatenated to one dataframe of length {len(surv_main_branch_df)} ###", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    '''
    ### Getting the main branch for all destroyed subhalos that merge to the main host halo: "subtrees"
    '''
    t_s_step = time.time()
    print("\n### Extracting the main branch ('subtree') for all destroyed subhalos that merge to the main host ... ", flush=True, file=out_f)
    # Full merger tree data of the host halo.
    host_halo_final_idx = host_main_branch['final.index'].iat[0]
    host_halo_full_tree_df = tree_grouped_by_final_tid.get_group(host_halo_final_idx)
    # Unique snapshot numbers of the host halo.
    host_unique_snapnums, host_unique_snapnum_idxs = np.unique(host_halo_full_tree_df['snapshot'].values, return_index=True)
    host_unique_snapnums_late_to_early = np.flip(host_unique_snapnums)
    host_unique_snapnum_idxs_late_to_early = np.flip(host_unique_snapnum_idxs)
    # A list to add subtree dataframes to: it will be converted to one big dataframe at the end.
    full_subtree_df_list = []
    num_subtrees = 0
    for i in range(len(host_unique_snapnums_late_to_early)):
        current_snap = host_unique_snapnums_late_to_early[i]
        # Main branch element of the host halo at the current snapshot
        current_snap_host_main_branch = host_main_branch.iloc[[i]]
        current_snap_host_index = current_snap_host_main_branch.index.values[0]
        # progenitor.main.last.dindex of all elements in host_halo_full_tree_df
        full_prog_main_last_didx_arr = host_halo_full_tree_df['progenitor.main.last.dindex'].to_numpy()
        if i == 0:
            print(f"\n  * Loop index: {i}, Last snapshot: no merging subhalos!  Host index: {current_snap_host_index}, # of host progenitors from previous snap: {current_snap_host_main_branch['progenitor.number'].to_numpy()[0]}", flush=True, file=out_f)
        elif i == (len(host_unique_snapnums_late_to_early)-1):
            print(f"  * Loop index: {i}, First snapshot: no merging subhalos!  Host index: {current_snap_host_index}", flush=True, file=out_f)
        else:
            print(f"  * Loop index: {i}, Snapshot: {current_snap}, Host index: {current_snap_host_index}, # of host progenitors from previous snap: {current_snap_host_main_branch['progenitor.number'].to_numpy()[0]}", flush=True, file=out_f)
            # Get all (sub)halos in the host full tree data at the current snapshot.
            # The number of halos in 'current_snap_subs' usually will not be the same as 'progenitor.number'.
            # because 'progenitor.number' only counts those that merge to the host halo at the current snapshot.
            current_snap_subs, first_idx, last_idx = host_tree_subhalos_one_snap_late_to_early(i, host_halo_full_tree_df, host_unique_snapnum_idxs_late_to_early)
            # Get indices (not tree indices, indices within current_snap_subs) of halos that merge to the host halo at the current snapshot.
            # Two ways to do this:
            #   1) Select those with 'descendant.index' equal to the host index at the next snapshot.
            #   2) Select thost that are NOT the main progenitor at the current snapshot: 'am.progenitor.main'==0.
            # Number of subhalos in subs_merging_now should be one fewer than the number of progenitors shown in the next snapshot element (current_snap_host_main_branch['progenitor.number'].values[0] at the next snapshot): 'progenitor.number' includes the main branch element.
            #descendant.index for current_snap_subs
            #descendant_idxs = current_snap_subs['descendant.index'].values
            #subs_merging_now = current_snap_subs.iloc[np.where(descendant_idxs==next_snap_host_idx)[0]]
            subs_merging_now = current_snap_subs.iloc[np.where(current_snap_subs['am.progenitor.main']==0)[0]]
            num_subtrees += len(subs_merging_now)
            # Extract main branches (subtrees) for all subhalos that are merging now: this step should be similar to how the main branches were obtained for surviving halos - gathering all elements, from the full tree data, that have the same 'progenitor.main.last.dindex'.
            # This works because the tree of these subhalos only stop being the main progenitor ('am.progenitor.main'=0) at the snapshot they merge to the main host.
            for j in range(len(subs_merging_now)):
                current_sub_prog_main_last_didx = subs_merging_now.iloc[[j]]['progenitor.main.last.dindex'].to_numpy()[0]
                current_sub_prog_idxs = np.where(full_prog_main_last_didx_arr == current_sub_prog_main_last_didx)[0]
                # Main branch (subtree) the current subhalo
                current_sub_tree = host_halo_full_tree_df.iloc[current_sub_prog_idxs]
                # Make sure the tree is correct by comparing the tree index and descendant.index.
                current_tree_desc_idxs = current_sub_tree['descendant.index'].to_numpy()
                current_tree_idxs = current_sub_tree.index.to_numpy()
                tree_tid = current_sub_tree['tid'].iat[0]
                if len(np.where(current_tree_idxs[:-1]!=current_tree_desc_idxs[1:])[0]) != 0:
                    msg = f"    *** ERROR: tree index and descendant.index not matching! This means wrong tree elements have been extracted as the main branch. tid-{tree_tid}, progenitor.main.last.dindex-{current_sub_prog_main_last_didx}"
                    print(msg, flush=True, file=out_f)
                    raise ValueError(msg)
                # Add a tree ID array to the current tree.
                tree_tid_arr = np.full(len(current_sub_tree), fill_value=tree_tid)
                current_sub_tree.insert(0, 'tree.tid', tree_tid_arr)
                # Append the subtree to the result list.
                full_subtree_df_list.append(current_sub_tree)
        next_snap_host_idx = current_snap_host_index # saving the host index from the current loop for the next loop: it will only be used if subs_merging_now uses 'descendant.index' instead of 'am.progenitor.main'.
    t_e_step = time.time()
    print("\n* Finished extracting the main branch ('subtree') for all destroyed subhalos that merge to the main host: ", end="", flush=True, file=out_f)
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Check if there's a scale factor column: if not, create one.
    col_names = full_subtree_df_list[0].columns
    matches = [col for col in possible_scale_col_names if col in col_names]
    if matches:
        # A scale factor column already exists: use its data, but change the name to 'scale.factor' if it's different.
        existing_col = matches[0]
        if existing_col != "scale.factor":
            for current_main_branch in full_subtree_df_list:
                current_main_branch["scale.factor"] = df.pop(existing_col) # rename
    else:
        t_s_step = time.time()
        # No scale factor column exists: create a new one.
        for current_main_branch in full_subtree_df_list:
            current_snapshot_arr = current_main_branch['snapshot'].to_numpy()
            snapshot_indices = np.isin(snap_z_a_flipped_df['snapshot_numbers'].values, current_snapshot_arr)
            # Scale factor array corresponding to the snapshot array.
            scale_arr = snap_z_a_flipped_df['scale_factors'].values[snapshot_indices]
            current_main_branch['scale.factor'] = scale_arr
        print("* scale.factor column added: ", end="", flush=True, file=out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Add distance.from.host.ckpc column.
    t_s_step = time.time()
    for i in range(len(full_subtree_df_list)):
        current_tree = full_subtree_df_list[i]
        current_tree_scales = current_tree['scale.factor'].to_numpy()
        # computing the distances using host.x, host.y, host.z, which are in physical kpc.
        current_tree_dist_arr = np.linalg.norm(current_tree[['host.x', 'host.y', 'host.z']].to_numpy(), None, 1) / current_tree_scales # ckpc
        '''
        # Below computes the distances by subtracting the coordinates of the host halo from the current halo. I have verified that the results are identical to just using ['host.x', 'host.y', 'host.z'], so I am using the simpler method.
        # Leaving it commented out in case ['host.x', 'host.y', 'host.z'] doesn't exist: it should always exist?
        current_tree_coords = current_tree[['x','y','z']].values
        #
        # Find the host tree's index corresponding to the first snapshot of the current tree.
        # Because the trees are ordered from latest to earliest snapshots, earliest_idx will be a larger number than latest_idx.
        host_tree_earliest_idx = np.where(np.isclose(host_scales, current_tree_scales[-1]))[0]
        idx = 1
        while len(host_tree_earliest_idx) == 0:
            # If the current tree begins before the first snapshot of the host tree, take the next snapshot, then the next and so on.
            host_tree_earliest_idx = np.where(np.isclose(host_scales, current_tree_scales[-1 - idx], atol=1e-4))[0]
            idx += 1
        host_tree_earliest_idx = host_tree_earliest_idx[0]
        # Find the host tree's index corresponding to the last snapshot of the current tree.
        host_tree_latest_idx = np.where(np.isclose(host_scales, current_tree_scales[0]))[0][0]
        # Make sure these indices correspond to the same snapshots as the first and last snapshots of the subtree.
        if current_tree_scales[0] != host_scales[host_tree_latest_idx]:
            print(f"*** Warning! scale factors between the host halo and the subtree DO NOT match!")
        if current_tree_scales[-1] != host_scales[host_tree_earliest_idx]:
            print(f"*** Warning! scale factors between the host halo and the subtree DO NOT match!")
        # Empty arrays of length = len(current_tree) instead of the number of selected snapshots because I want the result array to be of the same length as the tree.
        #current_tree_dist_arr = np.zeros(len(current_tree_scales))
        #if len(current_tree_scales) > len(host_scales):
        #    print(f"*** Warning! subtree is longer than the host halo main branch!")
        # 
        #print(host_scales[host_tree_latest_idx:host_tree_earliest_idx+1])
        #
        # Compute the halo distance at all selected snapshots.
        current_tree_dist_arr = np.linalg.norm(host_coords[host_tree_latest_idx:host_tree_earliest_idx+1] - current_tree_coords, None, 1) # ckpc, again, not sure about h
        '''
        # Add the distance array to the current dataframe.
        current_tree['distance.from.host.ckpc'] = current_tree_dist_arr
    print("* distance.from.host.ckpc column added: ", end="", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Convert the list of subtree dataframes to one big dataframe.
    t_s_step = time.time()
    dest_main_branch_df = pd.concat(full_subtree_df_list)
    t_e_step = time.time()
    print(f"### List of {len(full_subtree_df_list)} destroyed subhalo main branch (subtree) dataframes concatenated to one dataframe of length {len(dest_main_branch_df)} ###", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    '''
    * Creating a subhalo catalog using the merger tree data: this could be used as a comparison set for the subhalo catalog from Bloodhound.
      - Contains both surviving and destroyed subhalos
    '''
    t_s_step = time.time()
    print(f"\n### Creating a subhalo catalog using the merger tree data... ", flush=True, file=out_f)
    # Target time spacing for interpolation
    #target_t_space_gyr = 0.007
    target_t_space_gyr = BH_parameters['target_t_space_gyr']
    # Target time for savgol_filter window size
    #target_savgol_window_gyr = 0.2
    target_savgol_window_gyr = BH_parameters['target_savgol_window_gyr']
    # Radial velocity fraction consistency threshold before and after the pericenter.
    #vrad_sign_frac_threshold = 0.9
    vrad_sign_frac_threshold = BH_parameters['vrad_sign_frac_threshold']
    parameter_dict = peri_parameter_dict(target_t_space_gyr, target_savgol_window_gyr, vrad_sign_frac_threshold, host_main_branch['scale.factor'].to_numpy(), BH_parameters, out_f)
    #
    # Initialize the result catalog dictionary.
    subhalo_catalog_dict = {"surviving":{},
                            "destroyed":{}
                           }
    # Making the catalog for surviving subhalos.
    subhalo_type = "surviving"
    subhalo_catalog_dict = wrapper_subhalo_catalog_from_tree(host_main_branch, surv_main_branch_df, parameter_dict, subhalo_catalog_dict, subhalo_type, out_f)
    # Making the catalog for destroyed subhalos.
    subhalo_type = "destroyed"
    subhalo_catalog_dict = wrapper_subhalo_catalog_from_tree(host_main_branch, dest_main_branch_df, parameter_dict, subhalo_catalog_dict, subhalo_type, out_f)
    num_elem = len(subhalo_catalog_dict["surviving"]['ID.tree'])
    print(f"* Number of surviving subhalos from merger tree: {num_elem}", flush=True, file=out_f)
    num_elem = len(subhalo_catalog_dict["destroyed"]['ID.tree'])
    print(f"* Number of destroyed subhalos from merger tree: {num_elem}", flush=True, file=out_f)
    # Combine the surviving and destroyed subhalo catalogs to make one catalog.
    surv_sub_cat_df = pd.DataFrame.from_dict(subhalo_catalog_dict["surviving"], orient='columns')
    dest_sub_cat_df = pd.DataFrame.from_dict(subhalo_catalog_dict["destroyed"], orient='columns')
    sub_cat_df = pd.concat([surv_sub_cat_df, dest_sub_cat_df])
    t_e_step = time.time()
    print(f"### Subhalo catalog done! ###", flush=True, file=out_f)
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    '''
    * Saving output files
    '''
    # Save the surviving halo main branch data as a .hdf5 file.
    t_s_step = time.time()
    tree_processed_data_out_dir = BH_parameters['tree_processed_data_out_dir']
    surv_main_branch_fname = f"{tree_processed_data_out_dir}/main_branches.hdf5"
    surv_main_branch_df.to_hdf(surv_main_branch_fname, key='df', index=False)
    print(f"\n* Surviving halo main branch data saved at: {surv_main_branch_fname} ", end="", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Save the destroyed subhalo main branch data as a .hdf5 file.
    t_s_step = time.time()
    dest_main_branch_fname = f"{tree_processed_data_out_dir}/subtree_main_branches.hdf5"
    dest_main_branch_df.to_hdf(dest_main_branch_fname, key='df', index=False)
    print(f"* Destroyed subhalo main branch (subtree) data saved at: {dest_main_branch_fname} ", end="", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "*", True, out_f)
    # Save the main host halo main branch data as a .csv file.
    host_main_branch_fname = f"{tree_processed_data_out_dir}/host_main_branch.csv"
    host_main_branch.to_csv(host_main_branch_fname, index=False)
    print(f"* Main host halo main branch data saved at: {host_main_branch_fname} ", flush=True, file=out_f)
    # Save the subhalo catalog data as a .csv file.
    subhalo_catalog_fname = f"{tree_processed_data_out_dir}/subhalo_catalog.csv"
    sub_cat_df.to_csv(subhalo_catalog_fname, index=False)
    print(f"* Subhalo catalog data saved at: {subhalo_catalog_fname}", flush=True, file=out_f)
#
if __name__ == "__main__":
    main()