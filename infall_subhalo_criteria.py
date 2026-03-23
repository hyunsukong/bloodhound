'''
***** Load my environment first: conda activate my_env *****

* This script uses Consistent-trees merger tree data to find 
  subhaloes meeting the subhalo selection criteria, then
  make their subtree data.
* This, for now, is a stand-alone script. But eventually, I want to wrap it in
  the full Bloodhound script, run_bloodhound.py, that I haven't written yet.
'''
####################################################################################
# Import libraries.
####################################################################################
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
#import matplotlib
#from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import time
####################################################################################
# Import local modules/libraries/scripts.
####################################################################################
import halo_utilities
import halo_analysis
#import rockstar_handling
import utilities
#
####################################################################################
# Input parameters
####################################################################################
#parameter_fname = '/scratch/05097/hk9457/FIREII/m12b7e3_sidm1/bloodhound_subhalo_tracking/bloodhound_updating/BH_parameters/bloodhound_parameters.txt'
from config import parameter_fname
#
####################################################################################
# Functions
####################################################################################
'''
* Make file names for various process (sub)halo data.
- Make names only for files relevant for the subhalo selection criteria.
'''
def various_halo_file_names(base_dir, sim_num):
    # Initialize a dictionary to return with all file names.
    fname_dict = {}
    base_dir = f"{base_dir}/disk/halo_{sim_num}/subhalo_analysis"
    # Host main branch file.
    fname_dict["host_main_branch_dmo"] = f'{base_dir}/{sim_num}_dmo_host_main_branch_new.csv'
    fname_dict["host_main_branch_disk"] = f'{base_dir}/{sim_num}_disk_host_main_branch_new.csv'
    # (Destroyed) subtree main branch file.
    fname_dict["subtree_dmo"] = f"{base_dir}/{sim_num}_dmo_host_subtrees_main_branch_new.hdf5"
    fname_dict["subtree_disk"] = f"{base_dir}/{sim_num}_disk_host_subtrees_main_branch_new.hdf5"
    #
    return(fname_dict)
#
def various_halo_file_names_FIRE(base_dir, sim_num):
    # Initialize a dictionary to return with all file names.
    fname_dict = {}
    base_dir = f"{base_dir}/tree_processed_data"
    # Host main branch file name
    fname_dict["host_main_branch"] = f'{base_dir}/host_main_branch.csv'
    fname_dict["tree_main_branch"] = f'{base_dir}/main_branches.hdf5'
    fname_dict["subtree_main_branch"] = f'{base_dir}/subtree_main_branches.hdf5'
    #
    return(fname_dict)
'''
* Read-in the main branch data for the host halo.
'''
'''
def read_in_host_main_branch_file_FIRE(sim_file_name_dict, sim_num, BH_parameters):
    # All distance units will be converted to non-h units.
    h = BH_parameters['h']
    #
    # Make the dictionary key name for the subtree main branch file.
    fname_key = f"host_main_branch"
    #
    # Get the file name.
    fname = sim_file_name_dict[fname_key]
    #
    # Read in the host main branch file for the current simulation.
    host_main_branch_file = pd.read_csv(fname)
    #
    # Get various properties of the host tree as arrays.
    # Flip the arrays so they are ordered from early time to late time.
    host_x = np.flip(host_main_branch_file.x.values)
    host_y = np.flip(host_main_branch_file.y.values)
    host_z = np.flip(host_main_branch_file.z.values)
    host_snapnums = np.flip(host_main_branch_file.snapshot.values)
    host_vmax = np.flip(host_main_branch_file['vel.circ.max'].values)
    host_rvir = np.flip(host_main_branch_file.radius.values)
    #host_rvir_pkpc = np.multiply(host_rvir, host_scale)
    #host_tid = np.flip(host_main_branch_file.id.values)
    # Get the time information for the simulation.
    sim_snapshot_numbers = BH_parameters['time_info_dict']['snapshot_numbers']
    sim_scale_factors = BH_parameters['time_info_dict']['scale_factors']
    sim_redshifts = BH_parameters['time_info_dict']['redshifts']
    #sim_t_cosmic = BH_parameters['time_info_dict']['time']
    #sim_t_lookback = sim_t_cosmic[-1] - sim_t_cosmic
    # Match the snapshots to get the time information for the host halo tree.
    first_idx = np.nonzero(np.isclose(sim_snapshot_numbers, host_snapnums[0]))[0][0]
    host_scale = sim_scale_factors[first_idx:]
    host_redshift = sim_redshifts[first_idx:]
    #host_t_cosmic = sim_t_cosmic[first_idx:]
    #host_t_lookback = sim_t_lookback[first_idx:]
    #
    # Create a dictionary for the host halo data.
    host_dict = {}
    host_dict['x'] = host_x
    host_dict['y'] = host_y
    host_dict['z'] = host_z
    host_dict['scale.halo'] = host_scale
    host_dict['vmax'] = host_vmax
    host_dict['rvir'] = host_rvir
    #host_dict['rvir_phys'] = host_rvir_pkpc
    #host_dict['tree_id'] = host_tid
    return(host_dict)
'''
'''
#
def read_in_host_main_branch_file(sim_file_name_dict, sim_num, sim_type, BH_parameters):
    # All distance units will be converted to non-h units.
    h = BH_parameters['h']
    #
    # Make the dictionary key name for the subtree main branch file.
    #fname_key = f"host_main_branch_{sim_type}"
    fname_key = f"host_main_branch"
    #
    # Get the file name.
    fname = sim_file_name_dict[fname_key]
    #
    # Read in the host main branch file for the current simulation.
    host_main_branch_file = pd.read_csv(fname)
    #
    # Get various properties of the host tree as arrays.
    # Flip the arrays so they are ordered from early time to late time.
    host_x = np.flip(host_main_branch_file.x.values/h)
    host_y = np.flip(host_main_branch_file.y.values/h)
    host_z = np.flip(host_main_branch_file.z.values/h)
    host_scale = np.flip(host_main_branch_file.scale.values)
    host_redshift = 1. / host_scale - 1.
    host_t_cosmic = COSMO.age(host_redshift).value
    host_t_lookback = COSMO.lookback_time(host_redshift).value
    host_vmax = np.flip(host_main_branch_file.vmax.values)
    host_rvir = np.flip(host_main_branch_file.rvir.values/h)
    host_rvir_pkpc = np.multiply(host_rvir, host_scale)
    host_tid = np.flip(host_main_branch_file.id.values)
    #
    # Create a dictionary for the host halo data.
    host_dict = {}
    host_dict['x'] = host_x
    host_dict['y'] = host_y
    host_dict['z'] = host_z
    host_dict['scale.factor'] = host_scale
    host_dict['vmax'] = host_vmax
    host_dict['rvir'] = host_rvir
    host_dict['rvir_phys'] = host_rvir_pkpc
    host_dict['tree_id'] = host_tid
    return(host_dict)
'''
'''
* Read-in subtree main branch data.
  - Subtree data is the tree data of disrupted/merged subhalos of the host halo.
'''
def read_in_tree_main_branch_file_FIRE(sim_file_name_dict, tree_type):
    # Make the dictionary key name for the subtree main branch file.
    if tree_type == 'tree':
        fname_key = "tree_main_branch"
    elif tree_type == 'subtree':
        fname_key = "subtree_main_branch"
    #
    # Get the file name.
    fname = sim_file_name_dict[fname_key]
    #
    # Read in the subtree file for the current simulation.
    tree_df = pd.read_hdf(fname)
    return(tree_df)
#
def read_in_subtree_main_branch_file(sim_file_name_dict, sim_num, sim_type):
    # Make the dictionary key name for the subtree main branch file.
    fname_key = f"subtree_{sim_type}"
    #
    # Get the file name.
    fname = sim_file_name_dict[fname_key]
    #
    # Read in the subtree file for the current simulation.
    subtree_file = pd.read_hdf(fname)
    return(subtree_file)
'''
* Find broken-link tree subhalos from the subhalo selection dictionary.
- Then make two dictionaries containing:
    1) Infalling subhalos only,
    2) Broken-link tree subhalos only.
'''
def find_broken_link_tree_subhalos(infalling_subhalo_dict):
    # Initialize the result dictionary for broken-link tree subhalo data.
    # scale.factor.infall and idx.infall correspond to the first scale factor the broken link tree appears at, obviously not the infall scale factor.
    broken_link_subhalo_dict = {
        "tree.tid":[],
        "scale.factor.infall":[],
        "idx.infall":[],
        "subtree":[],
        "infalling?":[]
    }
    #
    # Identify indices for broken-link tree subhalos.
    # scale.factor.infall and idx.infall correspond to the first scale factor the broken link tree appears at, obviously not the infall scale factor.
    broken_link_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 0)[0]
    broken_link_subhalo_dict["tree.tid"] = infalling_subhalo_dict["tree.tid"][broken_link_subs_idx]
    broken_link_subhalo_dict["scale.factor.infall"] = infalling_subhalo_dict["scale.factor.infall"][broken_link_subs_idx]
    broken_link_subhalo_dict["idx.infall"] = infalling_subhalo_dict["idx.infall"][broken_link_subs_idx] # this might just contain a bunch of 0s?
    broken_link_subhalo_dict["subtree"] = infalling_subhalo_dict['subtree'][broken_link_subs_idx]
    broken_link_subhalo_dict["infalling?"] = infalling_subhalo_dict["infalling?"][broken_link_subs_idx]
    #
    # Identify indices for infalling subhalos.
    infalling_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 1)[0]
    infalling_subhalo_dict["tree.tid"] = infalling_subhalo_dict["tree.tid"][infalling_subs_idx]
    infalling_subhalo_dict["scale.factor.infall"] = infalling_subhalo_dict["scale.factor.infall"][infalling_subs_idx]
    infalling_subhalo_dict["idx.infall"] = infalling_subhalo_dict["idx.infall"][infalling_subs_idx]
    infalling_subhalo_dict["subtree"] = infalling_subhalo_dict["subtree"][infalling_subs_idx]
    infalling_subhalo_dict["infalling?"] = infalling_subhalo_dict["infalling?"][infalling_subs_idx]
    #
    # Return the result.
    return(infalling_subhalo_dict, broken_link_subhalo_dict)
#
def find_first_infall_from_subtree(subtree, host_scale_arr, host_rvir_arr):
    ###################
    ##### There is an almost identical, but newer function find_first_infall_from_tree() in halo_analysis.py
    ##### That one may be more up to date with column name conventions etc.
    ##### If I run into errors later with this function, I need to use the new one instead.
    '''
    * A function to find the first infall of a subhalo given the subtree for the subhalo found from the
      merger tree of the host halo.
    * This function will find the first time the subhalo is found inside the Rvir of the host halo, then
      1. will try to set the first infall scale factor as that corresponding to the snapshot immediately before
         when the subhalo is found inside for the first time.
      2. If the index is zero when the subhalo is found inside the Rvir for the first time,
         that scale factor will be the first infall scale factor.
    * Input:
      *** I could make this code to run in both early-to-late and late-to-early times. 
          But, for now, I assume that every input data is sorted from early to late times.
      - subtree:
      - host_scale_arr:
          - Full scale factor array for the host halo, in the same order as the scale factors in subtree.
      - host_rvir_arr:
          - Host's rvir in the same time ordering as host_scale_arr.

    * Returns:
      - Subtree's element index corresponding to the first infall scale factor
      - First infall scale factor
      - infalling: whether the subhalo actually falls in to the host halo or not.
        - 1: infalling
        - 0: not infalling - either not a subhalo or tree starts within the host halo.
    '''
    # Scale factor and distance for the subtree.
    sub_scale = subtree.scale.values
    sub_snaps = subtree['snapshot'].values
    sub_dist = subtree.dist.values
    #
    # Find indices to slice the data for the host.
    #host_first_idx = np.argmin(np.abs(host_scale_arr - scale_arr[0]))
    #host_last_idx = np.argmin(np.abs(host_scale_arr - scale_arr[-1]))
    #idx_1 = np.where(np.isclose(host_scale_arr, sub_scale[0], atol=5e-04))[0][0]
    #idx_2 = np.where(np.isclose(host_scale_arr, sub_scale[-1], atol=5e-04))[0][0]
    idx_1 = np.argmin(np.abs(host_scale_arr - sub_scale[0]))
    idx_2 = np.argmin(np.abs(host_scale_arr - sub_scale[-1]))
    # Host's rvir corresponding to the scale factors for the subtree.
    host_rvir_slice = host_rvir_arr[idx_1:idx_2+1]
    #print(sub_dist)
    #print(host_rvir_slice)
    # Compare subhalo's distance to host's rvir and find first infall.
    dist_subtracted = sub_dist - host_rvir_slice
    #
    # Find the first infall: first occurence of a negative value in the subtracted array.
    first_infall_idx = np.where(dist_subtracted < 0)[0]
    if len(first_infall_idx) > 0:
        # Subhalo has infall(s).
        if first_infall_idx[0] > 0:
            # If the first time the subhalo is found inside the Rvir is not the first index,
            # set the snapshot immediately before as the first infall snapshot and return its scale factor.
            infall_idx = first_infall_idx[0] - 1
            first_infall_scale = sub_scale[infall_idx]
            first_infall_snap = sub_snaps[infall_idx]
            infalling = 1
            #
            return(first_infall_scale, first_infall_snap, infall_idx, infalling)
        else:
            # Index is zero: the subtree is formed inside the Rvir of the host halo.
            # This is not an "infalling" subhalo, but keep its data.
            infall_idx = first_infall_idx[0]
            first_infall_scale = sub_scale[infall_idx]
            first_infall_snap = sub_snaps[infall_idx]
            infalling = 0
            return(first_infall_scale, first_infall_snap, infall_idx, infalling)
            #return(-1., -1, infalling)
        '''
                elif first_infall_idx[0] == 0 and sub_scale[0] <= 0.25:
            first_infall_scale = first_infall_idx[0]
            infall_idx = first_infall_idx[0]
            return(first_infall_scale, infall_idx)
        '''
    else:
        # Subhalo never falls within rvir of the host.
        return(-1., -1, -1, 0)
'''
* Find infalling subhalos with the first infall between z_high and z_low, and Vinfall > min_vinall
'''
def find_infalling_subhalos_FIRE(BH_parameters, main_branch_df, host_halo_dict, out_f):
    # Initialize the result dictionary.
    infalling_subhalo_dict = {"tree.tid":[],
                              "scale.factor.infall":[],
                              "snapshot.infall":[],
                              "idx.infall":[],
                              "subtree":[],
                              "infalling?":[],
                              "ID.halo.infall":[]}
    #
    # Read in relevant parameters: infall z, min/max V_infall.
    low_z = BH_parameters['first_infall_z_low']
    high_z = BH_parameters['first_infall_z_high']
    low_z_scale = 1./(low_z + 1.)
    high_z_scale = 1./(high_z + 1.)
    #
    # Snapshot numbers corresponding to the redshift thresholds.
    sim_snapshot_numbers = BH_parameters['time_info_dict']['snapshot_numbers']
    sim_scale_factors = BH_parameters['time_info_dict']['scale_factors']
    sim_redshifts = BH_parameters['time_info_dict']['redshifts']
    low_z_idx = np.argmin(np.abs(sim_scale_factors - low_z_scale))
    high_z_idx = np.argmin(np.abs(sim_scale_factors - high_z_scale))
    #low_z_idx = np.nonzero(np.isclose(sim_scale_factors, low_z_scale, atol=1e-4))[0][0]
    #high_z_idx = np.nonzero(np.isclose(sim_scale_factors, high_z_scale, atol=1e-4))[0][0]
    low_z_snap = sim_snapshot_numbers[low_z_idx]
    high_z_snap = sim_snapshot_numbers[high_z_idx]
    min_vinfall = BH_parameters['min_vinfall']
    max_vinfall = BH_parameters['max_vinfall']
    #
    # Host halo information: arrays are sorted from early to late times, same as the subtree data.
    host_scale_arr = host_halo_dict['scale.factor']
    host_rvir_arr = host_halo_dict['rvir'] # Physical kpc, might be COMOVING!
    host_x_arr = host_halo_dict['x'] * host_scale_arr # Physical kpc
    host_y_arr = host_halo_dict['y'] * host_scale_arr
    host_z_arr = host_halo_dict['z'] * host_scale_arr
    host_coords_arr = np.column_stack((host_x_arr, host_y_arr, host_z_arr)) # Physical kpc
    #
    # Group the subtree main branch dataframe by the subtree ID.
    main_branch_grouped = main_branch_df.groupby('tree.tid')
    print(f"* Total length of the tree main branch dataframe: {len(main_branch_df)} rows", flush=True, file=out_f)
    print(f"* Total number of trees: {len(main_branch_grouped)}", flush=True, file=out_f)
    #
    # Find all subtrees that had min_vinfall<=Vmax<max_vinfall for at least one snapshot between low_z<=z<=high_z.
    #vmax_query = subtree_main_branch_df.query('@min_vinfall <= `vel.circ.max` < @max_vinfall & @high_z_scale <= scale <= @low_z_scale')
    # Using snapshot numbers instead.
    #vmax_query = main_branch_df.query('@min_vinfall <= `vel.circ.max` < @max_vinfall & @high_z_snap <= snapshot <= @low_z_snap')
    #vmax_query_uniq_sids = np.unique(vmax_query['tree.tid'].values)
    #out_string = (
    #    f"* Number of subtrees that had {min_vinfall}<=Vmax<{max_vinfall}km/s"
    #    f" at least once between {low_z}<=z<={high_z}: {len(vmax_query_uniq_sids)}"
    #)
    #print(out_string, flush=True, file=out_f)
    #
    # Loop through each subtree in vmax_query and identify subhalos that meet the infall criteria.
    uniq_sids = np.unique(main_branch_df['tree.tid'].values)
    #for i in range(len(vmax_query_uniq_sids)):
    for i in range(len(uniq_sids)):
        #current_sid = vmax_query_uniq_sids[i]
        current_sid = uniq_sids[i]
        #current_subtree = main_branch_grouped.get_group(current_sid)[::-1] # Order the subtree from early to late times.
        current_subtree = main_branch_grouped.get_group(current_sid) # ordered from late to early times.
        '''
        # Don't need these anymore because the updated tree_pre_processing part already inserts scale factor and distance information
        #
        # Add the scale factor array to the dataframe.
        first_snap = current_subtree.iloc[0].snapshot
        last_snap = current_subtree.iloc[-1].snapshot
        first_snap_idx = np.nonzero(np.isclose(sim_snapshot_numbers, first_snap))[0][0]
        last_snap_idx = np.nonzero(np.isclose(sim_snapshot_numbers, last_snap))[0][0]
        scale_arr = sim_scale_factors[first_snap_idx:last_snap_idx+1]
        current_subtree.insert(1, 'scale', scale_arr)
        # Add the distance (wrt host center) array to the dataframe.
        x_arr = current_subtree['x'].values * scale_arr # Physical kpc
        y_arr = current_subtree['y'].values * scale_arr
        z_arr = current_subtree['z'].values * scale_arr
        coords_arr = np.column_stack((x_arr, y_arr, z_arr)) # Physical kpc
        # host's coordinates at the snapshots of the subhalo elements.
        #low_z_idx = np.argmin(np.abs(sim_scale_factors - low_z_scale))
        #high_z_idx = np.argmin(np.abs(sim_scale_factors - high_z_scale))
        #host_first_idx = np.nonzero(np.isclose(host_scale_arr, scale_arr[0], atol=1e-4))[0][0]
        #host_last_idx = np.nonzero(np.isclose(host_scale_arr, scale_arr[-1], atol=1e-4))[0][0]
        host_first_idx = np.argmin(np.abs(host_scale_arr - scale_arr[0]))
        host_last_idx = np.argmin(np.abs(host_scale_arr - scale_arr[-1]))
        host_coords_arr_use = host_coords_arr[host_first_idx:host_last_idx+1]
        dist_arr = np.linalg.norm(coords_arr - host_coords_arr_use, None, 1)
        current_subtree.insert(2, 'dist', dist_arr)
        '''
        #
        # Find the first infall scale factor for the current subhalo.
        #infall_a, infall_snap, infall_idx, infalling = find_first_infall_from_subtree(current_subtree, host_scale_arr, host_rvir_arr)
        infall_a, infall_idx, infalling = halo_analysis.find_first_infall_from_tree(current_subtree, host_scale_arr, host_rvir_arr)
        #
        # Check if the first infall is between low_z<=z<=high_z.
        if high_z_scale <= infall_a <= low_z_scale:
            # Take the tree element at the infall scale factor.
            infall_row = current_subtree.iloc[infall_idx]
            infall_snap = int(infall_row['snapshot'])
            # Check the fraction of low-resolution mass contamination, and remove if it above the threshold given by the parameter file.
            low_res_frac = BH_parameters['low_res_frac']
            mass_ratio = infall_row['mass.lowres']/ infall_row['mass.vir']
            if mass_ratio > low_res_frac:
                continue
            catalog_ID_at_infall = int(infall_row['catalog.index'])
            if catalog_ID_at_infall < 0:
                # Halo at the infall snapshot is a phantom halo, so replace the infall snapshot with an nearby snapshot that has the highest value of 'vel.circ.max'.
                # Halo at the infall snapshot is a phantom halo, so replace the infall snapshot with the nearest non-phantom row.
                non_phantom_rows = current_subtree.query('`catalog.index`>0')
                if len(non_phantom_rows) == 0:
                    print(f"*** {current_sid}: all tree entries are phantom! You should never see this message!", flush=True, file=out_f)
                else:
                    non_phantom_snapnums = non_phantom_rows['snapshot'].values
                    nearest_idx = np.argmin(np.abs(non_phantom_snapnums - infall_snap))
                    infall_row = non_phantom_rows.iloc[nearest_idx]
                    idx_difference = int(infall_snap) - int(infall_row['snapshot'])
                    infall_idx = int(infall_idx - idx_difference)
                    infall_a = infall_row['scale.factor']
                    infall_snap = int(infall_row['snapshot'])
                    catalog_ID_at_infall = int(infall_row['catalog.index'])
            #
            # Check if min_vinfall<=Vinfall<max_vinfall.
            #if infall_row.vmax>=min_vinfall:
            if infall_row['vel.circ.max']>=min_vinfall and infall_row['vel.circ.max']<=max_vinfall:
                infalling_subhalo_dict["tree.tid"].append(current_sid)
                infalling_subhalo_dict["scale.factor.infall"].append(infall_a)
                infalling_subhalo_dict["snapshot.infall"].append(infall_snap)
                infalling_subhalo_dict["idx.infall"].append(infall_idx)
                infalling_subhalo_dict["subtree"].append(current_subtree)
                infalling_subhalo_dict["infalling?"].append(infalling)
                infalling_subhalo_dict["ID.halo.infall"].append(catalog_ID_at_infall)
    #
    # Convert lists to arrays.
    infalling_subhalo_dict["tree.tid"] = np.array(infalling_subhalo_dict["tree.tid"])
    infalling_subhalo_dict["scale.factor.infall"] = np.array(infalling_subhalo_dict["scale.factor.infall"])
    infalling_subhalo_dict["snapshot.infall"] = np.array(infalling_subhalo_dict["snapshot.infall"])
    infalling_subhalo_dict["idx.infall"] = np.array(infalling_subhalo_dict["idx.infall"])
    infalling_subhalo_dict["subtree"] = np.array(infalling_subhalo_dict["subtree"], dtype=object)
    infalling_subhalo_dict["infalling?"] = np.array(infalling_subhalo_dict["infalling?"])
    infalling_subhalo_dict["ID.halo.infall"] = np.array(infalling_subhalo_dict["ID.halo.infall"])
    #
    # Sort the infalling subhalo dictionary data by the infall scale factor.
    infall_scale_argsort_idx = np.argsort(infalling_subhalo_dict["scale.factor.infall"])
    infalling_subhalo_dict["tree.tid"] = infalling_subhalo_dict["tree.tid"][infall_scale_argsort_idx]
    infalling_subhalo_dict["scale.factor.infall"] = infalling_subhalo_dict["scale.factor.infall"][infall_scale_argsort_idx]
    infalling_subhalo_dict["snapshot.infall"] = infalling_subhalo_dict["snapshot.infall"][infall_scale_argsort_idx]
    infalling_subhalo_dict["idx.infall"] = infalling_subhalo_dict["idx.infall"][infall_scale_argsort_idx]
    infalling_subhalo_dict["subtree"] = infalling_subhalo_dict["subtree"][infall_scale_argsort_idx]
    infalling_subhalo_dict["infalling?"] = infalling_subhalo_dict["infalling?"][infall_scale_argsort_idx]
    infalling_subhalo_dict["ID.halo.infall"] = infalling_subhalo_dict["ID.halo.infall"][infall_scale_argsort_idx]
    #
    out_string = (
        f"* Number of subhalos with {low_z}<=z_infall<={high_z}"
        f" and {min_vinfall}<=V_infall<{max_vinfall}km/s:"
        f" {len(infalling_subhalo_dict['tree.tid'])}"
    )
    print(out_string, flush=True, file=out_f)
    #
    # Return results.
    return(infalling_subhalo_dict)
def find_infalling_subhalos(BH_parameters, subtree_main_branch_df, host_halo_dict, out_f):
    # Initialize the result dictionary.
    infalling_subhalo_dict = {"tree.tid":[],
                              "scale.factor.infall":[],
                              "idx.infall":[],
                              "subtree":[],
                              "infalling?":[]}
    #
    # Read in relevant parameters: infall z, min/max V_infall.
    low_z = BH_parameters['first_infall_z_low']
    high_z = BH_parameters['first_infall_z_high']
    low_z_scale = 1./(low_z + 1.)
    high_z_scale = 1./(high_z + 1.)
    min_vinfall = BH_parameters['min_vinfall']
    max_vinfall = BH_parameters['max_vinfall']
    #
    # Host halo information: arrays are sorted from early to late times, same as the subtree data.
    # Also, I am pretty sure the subhalo distance and host rvir arrays are in comoving units.
    host_scale_arr = host_halo_dict['scale.factor']
    host_rvir_arr = host_halo_dict['rvir']
    #
    # Group the subtree main branch dataframe by the subtree ID.
    subtree_main_branch_grouped = subtree_main_branch_df.groupby('subtree_id')
    print(f"* Total length of the subtree main branch dataframe: {len(subtree_main_branch_df)} rows", flush=True, file=out_f)
    print(f"* Total number of subtrees: {len(subtree_main_branch_grouped)}", flush=True, file=out_f)
    #
    # Find all subtrees that had min_vinfall<=Vmax<max_vinfall for at least one snapshot between low_z<=z<=high_z.
    vmax_query = subtree_main_branch_df.query('@min_vinfall <= vmax < @max_vinfall & @high_z_scale <= scale <= @low_z_scale')
    vmax_query_uniq_sids = np.unique(vmax_query.subtree_id.values)
    out_string = (
        f"* Number of subtrees that had {min_vinfall}<=Vmax<{max_vinfall}km/s"
        f" at least once between {low_z}<=z<={high_z}: {len(vmax_query_uniq_sids)}"
    )
    print(out_string, flush=True, file=out_f)
    #
    # Loop through each subtree in vmax_query and identify subhalos that meet the infall criteria.
    for i in range(len(vmax_query_uniq_sids)):
        current_sid = vmax_query_uniq_sids[i]
        current_subtree = subtree_main_branch_grouped.get_group(current_sid)[::-1] # Order the subtree from early to late times.
        #
        # Find the first infall scale factor for the current subhalo.
        infall_a, infall_idx, infalling = halo_analysis.find_first_infall_from_subtree(current_subtree, host_scale_arr, host_rvir_arr)
        #
        # Check if the first infall is between low_z<=z<=high_z.
        if high_z_scale <= infall_a <= low_z_scale:
            # Take the tree element at the infall scale factor.
            infall_row = current_subtree.iloc[infall_idx]
            #
            # Check if min_vinfall<=Vinfall<max_vinfall.
            #if infall_row.vmax>=min_vinfall:
            if infall_row.vmax>=min_vinfall and infall_row.vmax<=max_vinfall:
                infalling_subhalo_dict["tree.tid"].append(current_sid)
                infalling_subhalo_dict["scale.factor.infall"].append(infall_a)
                infalling_subhalo_dict["idx.infall"].append(infall_idx)
                infalling_subhalo_dict["subtree"].append(current_subtree)
                infalling_subhalo_dict["infalling?"].append(infalling)
    #
    # Convert lists to arrays.
    infalling_subhalo_dict["tree.tid"] = np.array(infalling_subhalo_dict["tree.tid"])
    infalling_subhalo_dict["scale.factor.infall"] = np.array(infalling_subhalo_dict["scale.factor.infall"])
    infalling_subhalo_dict["idx.infall"] = np.array(infalling_subhalo_dict["idx.infall"])
    infalling_subhalo_dict["subtree"] = np.array(infalling_subhalo_dict["subtree"], dtype=object)
    infalling_subhalo_dict["infalling?"] = np.array(infalling_subhalo_dict["infalling?"])
    #
    # Sort the infalling subhalo dictionary data by the infall scale factor.
    infall_scale_argsort_idx = np.argsort(infalling_subhalo_dict["scale.factor.infall"])
    infalling_subhalo_dict["tree.tid"] = infalling_subhalo_dict["tree.tid"][infall_scale_argsort_idx]
    infalling_subhalo_dict["scale.factor.infall"] = infalling_subhalo_dict["scale.factor.infall"][infall_scale_argsort_idx]
    infalling_subhalo_dict["idx.infall"] = infalling_subhalo_dict["idx.infall"][infall_scale_argsort_idx]
    infalling_subhalo_dict["subtree"] = infalling_subhalo_dict["subtree"][infall_scale_argsort_idx]
    infalling_subhalo_dict["infalling?"] = infalling_subhalo_dict["infalling?"][infall_scale_argsort_idx]
    #
    out_string = (
        f"* Number of subhalos with {low_z}<=z_infall<={high_z}"
        f" and {min_vinfall}<=V_infall<{max_vinfall}km/s:"
        f" {len(infalling_subhalo_dict['tree.tid'])}"
    )
    print(out_string, flush=True, file=out_f)
    #
    # Return results.
    return(infalling_subhalo_dict)

'''
* A function to connect a single halo found from one set of Rockstar+consistent-trees to another set of Rockstar results.
  - Function connect_rockstar_sets() uses this function.
'''
def find_halo_in_catalog(current_halo_df, halo_catalog_df, com_range, vmax_range, h, out_f, j):
    # Coordinates and Vmax of the current halo to use to find the match.
    tree_coords = current_halo_df[['x', 'y', 'z']].values[0] / h # Non-h ckpc
    tree_vmax = current_halo_df['vmax'].values[0]
    #
    # Find all subhalos in the catalog that are within the com_range of the COM of the current halo.
    cat_halos_within_com = halo_utilities.match_halo_to_catalog_com(tree_coords, halo_catalog_df, com_range)
    #
    if len(cat_halos_within_com) == 0:
        # There are no halos in the catalog within com_range of the COM of the current halo.
        print(f"  * No COM match: subhalo index-{j}, subtree ID-{current_halo_df['subtree_id'].values[0]}, Vinfall-{tree_vmax} km/s", flush=True, file=out_f)
        #
        # Set -1 as a place-holder result.
        com_match_halo = -1
    elif len(cat_halos_within_com) > 1:
        # More than one halos within com_range.
        matched_halos_vmaxs = cat_halos_within_com.vmax.values
        matched_halos_hids = cat_halos_within_com.orig_id.values
        print(f"  * More than one COM match: subhalo index-{j}, subtree ID-{current_halo_df['subtree_id'].values[0]}, Vinfall-{tree_vmax} km/s, matched halo ID-{matched_halos_hids}, matched Vmax-{matched_halos_vmaxs} km/s", flush=True, file=out_f)
        #
        # Use Vmax to find the correct COM match.
        closest_vmax_idx = np.argmin(np.absolute(matched_halos_vmaxs - tree_vmax))
        com_match_halo = cat_halos_within_com.iloc[[closest_vmax_idx]] # Double bracket gives a dataframe instead of a series.
        print(f"    * Closest Vinfall match: halo ID-{matched_halos_hids}, Vmax-{com_match_halo.vmax.values[0]} km/s", flush=True, file=out_f)
    else:
        # There is exactly one halo within com_range.
        com_match_halo = cat_halos_within_com
    #
    # For COM matched halo, check if its Vmax value from the second set of catalogs is within vmax_range of the Vinfall from the tree.
    if isinstance(com_match_halo, int) != True: # Skipping com_match_halo=-1
        matched_halo_vmax = com_match_halo.vmax.values[0]
        if np.absolute(matched_halo_vmax - tree_vmax)/tree_vmax > vmax_range:
            #
            # Set -1 as a place-holder result.
            com_match_halo = -1
            print(f"  * Vmax value of the matched halo is not within {vmax_range * 100} %, so there is no matching halo: ", flush=True, file=out_f)
            print(f"    * subhalo index-{j}, subtree ID-{current_halo_df['subtree_id'].values[0]}, Vinfall-{tree_vmax} km/s, matched Vmax-{matched_halo_vmax} km/s", flush=True, file=out_f)
    #
    return(com_match_halo)
'''
* A function to connect halos found from one set of Rockstar+consistent-trees to another set of Rockstar results.
  - This function assumes the infalling subhalo dictionary data is sorted by the infall scale factor.
  - This function updates the existing input infall_subhalo_dict dictionary!
'''
def connect_rockstar_sets(sim_num, BH_parameters, snapnum_info_dict, infalling_subhalo_dict, out_f):
    # Add additional dictionary keys to the existing infalling_subhalo_dict dictionary.
    infalling_subhalo_dict["ID.halo.infall"]=[]
    infalling_subhalo_dict["snapshot.infall"]=[]
    #
    # COM and Vmax ranges to use for the search
    com_range = BH_parameters['com_range']
    vmax_range = BH_parameters['vmax_range']
    print(f"* Search range for COM coordinates: {com_range} physical Mpc", flush=True, file=out_f)
    print(f"* Search range for Vmax: {vmax_range * 100} %", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    #
    # Read in relevant parameters from the parameter dictionary.
    # All distance units will be converted to non-h units.
    h = BH_parameters['h']
    base_dir = BH_parameters['base_dir']
    run_type = BH_parameters['run_type']
    #
    full_snapnum_arr = snapnum_info_dict['snapshot_numbers']
    full_a_arr = snapnum_info_dict['scale_factors']
    #
    # Make a infall scale factor array with no repeated values.
    unique_infall_scale_arr, counts = np.unique(infalling_subhalo_dict["scale.factor.infall"], return_counts=True)
    start_idx = 0
    for i in range(len(unique_infall_scale_arr)):
        current_scale = unique_infall_scale_arr[i]
        com_range_cMpc = com_range / current_scale # com_range in comoving Mpc
        num_subs = counts[i]
        end_idx = start_idx + num_subs
        #
        # Get the infalling subhalo data for the current scale factor.
        current_scale_sids = infalling_subhalo_dict['tree.tid'][start_idx:end_idx]
        current_scale_infall_idxs = infalling_subhalo_dict['idx.infall'][start_idx:end_idx]
        current_scale_infall_scales = infalling_subhalo_dict['scale.factor.infall'][start_idx:end_idx]
        current_scale_infall_subtrees = infalling_subhalo_dict['subtree'][start_idx:end_idx]
        #
        # A check to make sure that all elements in current_scale_infall_scales is equal to current_scale.
        if len(np.where(current_scale_infall_scales != current_scale)[0]) != 0:
            print("", flush=True, file=out_f)
            print("*** Error! *** ", flush=True, file=out_f)
            print("* There is an incorrect infall scale factor in the infall scale factor array:", flush=True, file=out_f)
            print(f"* Current infall scale factor: {current_scale}", flush=True, file=out_f)
            print(f"* Array index of the wrong value: {np.where(current_scale_infall_scales != current_scale)[0]}", flush=True, file=out_f)
            print("", flush=True, file=out_f)
        #
        # Get the snapshot number corresponding to the current scale factor from the snapshot information dictionary.
        # Here, np.nonzero could be used in place of np.where.
        snap_idx = np.where(np.isclose(full_a_arr, current_scale, atol=1e-04))[0][0]
        current_snap = full_snapnum_arr[snap_idx]
        #infalling_subhalo_dict["snapshot.infall"].append(current_snap)
        print(f"* Matching {num_subs} subhalos at a={current_scale} (snapshot={current_snap})...", flush=True, file=out_f)
        #
        # Make the file name for the halo catalog to match the halo from.
        cat_fname = f"{base_dir}/{run_type}/halo_{sim_num}/halo_catalogs/{sim_num}_{run_type}_catalog_snap_{current_snap:03}.csv"
        #
        # Read in the catalog file.
        current_catalog = pd.read_csv(cat_fname)
        #
        # For the current infall scale factor, for each subhalo found from one set of Rockstar/consistent-trees,
        # find the corresponding subhalo from the other set.
        for j in range(len(current_scale_sids)):
            current_sid = current_scale_sids[j]
            current_sub_infall_idx = current_scale_infall_idxs[j]
            #
            # Take the subtree element for the current subhalo at its infall snapshot.
            current_subtree_at_infall = current_scale_infall_subtrees[j].iloc[[current_sub_infall_idx]]
            #
            # Find a matching halo from the other halo catalog.
            # matched_halo will have a pandas.df element of the matched halo, otherwise set to 0.
            current_sub_matched_halo = find_halo_in_catalog(current_subtree_at_infall, current_catalog, com_range_cMpc, vmax_range, h, out_f, j)
            #
            # Append the matched halo's halo ID to the result dictionary.
            if isinstance(current_sub_matched_halo, int) != True:
                # Use orig_id instead of halo_id because halo_id is the new ID I set using the row number for the Vmax>4.5 km/s catalog.
                matched_hID = current_sub_matched_halo.orig_id.values[0]
            else:
                matched_hID = -1
            infalling_subhalo_dict["ID.halo.infall"].append(matched_hID)
            #
            # Append the infall snapshot number to the result dictionary.
            infalling_subhalo_dict["snapshot.infall"].append(current_snap)
        #
        # Set the new start index.
        start_idx = end_idx
    #
    # Convert lists to arrays.
    infalling_subhalo_dict["ID.halo.infall"] = np.array(infalling_subhalo_dict["ID.halo.infall"])
    infalling_subhalo_dict["snapshot.infall"] = np.array(infalling_subhalo_dict["snapshot.infall"])
    return(infalling_subhalo_dict)
#
def summary_statement_FIRE(infalling_subhalo_dict, out_f):
    print("### Infalling subhalo identification finished! ###", flush=True, file=out_f)
    print("* The result dictionary has the following keys:", flush=True, file=out_f)
    print(f"  * {list(infalling_subhalo_dict.keys())}", flush=True, file=out_f)
    #
    # Identify indices for actually infalling subhalos and brokin-link subhalos.
    infalling_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 1)[0]
    broken_link_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 0)[0]
    #
    num_all_infall_subs = len(infalling_subhalo_dict['tree.tid'])
    num_real_infall_subs = len(infalling_subs_idx)
    num_broken_link_subs = len(broken_link_subs_idx)
    #
    print(f"* Number of subhalos meeting 1) infall time and 2) Vinfall criteria: {num_all_infall_subs}", flush=True, file=out_f)
    print(f"* Number of broken-link tree subhalos: {num_broken_link_subs}", flush=True, file=out_f)
    print(f"* Number of actually infalling subhalos: {num_real_infall_subs}", flush=True, file=out_f)
#
def summary_statement(infalling_subhalo_dict, out_f):
    print("### Infalling subhalo identification finished! ###", flush=True, file=out_f)
    print("* The result dictionary has the following keys:", flush=True, file=out_f)
    print(f"  * {list(infalling_subhalo_dict.keys())}", flush=True, file=out_f)
    #
    # Identify indices for actually infalling subhalos and brokin-link subhalos.
    infalling_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 1)[0]
    broken_link_subs_idx = np.where(infalling_subhalo_dict["infalling?"] == 0)[0]
    #
    num_all_infall_subs = len(infalling_subhalo_dict['tree.tid'])
    num_all_infall_with_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"] != -1)[0])
    num_all_infall_without_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"] == -1)[0])
    num_broken_link_subs = len(broken_link_subs_idx)
    num_broken_link_with_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"][broken_link_subs_idx] != -1)[0])
    num_broken_link_without_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"][broken_link_subs_idx] == -1)[0])
    num_real_infall_subs = len(infalling_subs_idx)
    num_real_infall_with_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"][infalling_subs_idx] != -1)[0])
    num_real_infall_without_match = len(np.where(infalling_subhalo_dict["ID.halo.infall"][infalling_subs_idx] == -1)[0])
    #
    print(f"* Number of subhalos meeting 1) infall time and 2) Vinfall criteria: {num_all_infall_subs}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITH a match from the other set of Rockstar catalogs: {num_all_infall_with_match}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITHOUT a match from the other set of Rockstar catalogs: {num_all_infall_without_match}", flush=True, file=out_f)
    print(f"* Number of broken-link tree subhalos: {num_broken_link_subs}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITH a match from the other set of Rockstar catalogs: {num_broken_link_with_match}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITHOUT a match from the other set of Rockstar catalogs: {num_broken_link_without_match}", flush=True, file=out_f)
    print(f"* Number of actually infalling subhalos: {num_real_infall_subs}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITH a match from the other set of Rockstar catalogs: {num_real_infall_with_match}", flush=True, file=out_f)
    print(f"  * Number of subhalos WITHOUT a match from the other set of Rockstar catalogs: {num_real_infall_without_match}", flush=True, file=out_f)
    print("", flush=True, file=out_f)
'''
* A function to save the subhalo infall criteria result subtrees.
- Before saving, infall scale factor and infall snapshot number data will be added to the existing subtrees.
- Also, the halo ID of the matched halo from the other Rockstar set will be added.
  - For a subhalo without a match, -1 will be added.
  - So if the initial Rockstar set had the particle data saved, the entire column should be -1.
- Also, infalling? column will be added.
'''
def save_infall_subtree_result(infalling_subhalo_dict, BH_parameters, sim_num, sim_type, out_f):
    subtree_arr = infalling_subhalo_dict["subtree"]
    infall_scale_factor_arr = infalling_subhalo_dict["scale.factor.infall"]
    infall_snapnum_arr = infalling_subhalo_dict["snapshot.infall"]
    matched_hID_arr = infalling_subhalo_dict["ID.halo.infall"]
    infall_check_arr = infalling_subhalo_dict["infalling?"]
    #
    # Add infall scale factor, infall snapshot number, and matched halo ID columns to each subtree.
    for i in range(len(subtree_arr)):
        current_tree = subtree_arr[i]
        current_tree_infall_scale = infall_scale_factor_arr[i]
        current_tree_infall_snapnum = infall_snapnum_arr[i]
        current_tree_matched_hID = matched_hID_arr[i]
        current_tree_infall_check = infall_check_arr[i]
        #
        # Number of dataframe rows for the current subtree.
        df_length = len(current_tree)
        #
        # Make the infall scale factor, infall snapshot number, matched halo ID, and infall check arrays.
        current_tree_infall_scale_arr = np.full(df_length, current_tree_infall_scale)
        current_tree_infall_snapnum_arr = np.full(df_length, current_tree_infall_snapnum)
        current_tree_matched_hID_arr = np.full(df_length, current_tree_matched_hID)
        current_tree_infall_check_arr = np.full(df_length, current_tree_infall_check)
        #
        # Append the infall scale factor, infall snapshot number, matched halo ID, and infall check arrays as dataframe columns.
        current_tree['scale.factor.infall'] = current_tree_infall_scale_arr
        current_tree['snapshot.infall'] = current_tree_infall_snapnum_arr
        current_tree['ID.halo.infall'] = current_tree_matched_hID_arr
        current_tree['infalling?'] = current_tree_infall_check_arr
    #
    # Combine the array of dataframes as one big dataframe of subtrees.
    combined_subtree_df = pd.concat(subtree_arr)
    #
    # Make the output file name.
    out_dir = BH_parameters['infall_subtree_out_dir']
    out_fname_base = BH_parameters['infall_subtree_out_fname_base']
    out_fname = f"{out_dir}/{sim_num}_{sim_type}_{out_fname_base}.hdf5"
    #
    # Save the output file.
    combined_subtree_df.to_hdf(out_fname, key='df', mode='w')
    print(f"* File path: {out_fname}", flush=True, file=out_f)
#
def save_infall_subtree_result_FIRE(infalling_subhalo_dict, BH_parameters, sim_num, tree_type, out_f):
    subtree_arr = infalling_subhalo_dict["subtree"]
    infall_scale_factor_arr = infalling_subhalo_dict["scale.factor.infall"]
    infall_snapnum_arr = infalling_subhalo_dict["snapshot.infall"]
    catalog_hID_arr = infalling_subhalo_dict["ID.halo.infall"]
    infall_check_arr = infalling_subhalo_dict["infalling?"]
    #
    # Add infall scale factor, infall snapshot number, and matched halo ID columns to each subtree.
    for i in range(len(subtree_arr)):
        current_tree = subtree_arr[i]
        current_tree_infall_scale = infall_scale_factor_arr[i]
        current_tree_infall_snapnum = infall_snapnum_arr[i]
        current_tree_infall_check = infall_check_arr[i]
        current_tree_catalog_hID = catalog_hID_arr[i]
        #
        # Number of dataframe rows for the current subtree.
        df_length = len(current_tree)
        #
        # Make the infall scale factor, infall snapshot number, matched halo ID, and infall check arrays.
        current_tree_infall_scale_arr = np.full(df_length, current_tree_infall_scale)
        current_tree_infall_snapnum_arr = np.full(df_length, current_tree_infall_snapnum)
        current_tree_infall_check_arr = np.full(df_length, current_tree_infall_check)
        current_tree_catalog_hID_arr = np.full(df_length, current_tree_catalog_hID)
        #
        # Append the infall scale factor, infall snapshot number, matched halo ID, and infall check arrays as dataframe columns.
        current_tree['scale.factor.infall'] = current_tree_infall_scale_arr
        current_tree['snapshot.infall'] = current_tree_infall_snapnum_arr
        current_tree['infalling?'] = current_tree_infall_check_arr
        current_tree['ID.halo.infall'] = current_tree_catalog_hID_arr
    #
    # Combine the array of dataframes as one big dataframe of subtrees.
    combined_subtree_df = pd.concat(subtree_arr)
    #
    # Make the output file name.
    out_dir = BH_parameters['infall_subtree_out_dir']
    out_fname_base = BH_parameters['infall_subtree_out_fname_base']
    out_fname = f"{out_dir}/{sim_num}_{tree_type}_{out_fname_base}.hdf5"
    #
    # Save the output file.
    combined_subtree_df.to_hdf(out_fname, key='df', mode='w')
    print(f"* File path: {out_fname}", flush=True, file=out_f)
'''
* 'Main' wrapper function for finding infall subhalos.
- For one simulation.
- I have not gotten around to making the code to handle the case where one does NOT need to connect one set of Rockstar/consistent-trees
  result to another set of Rockstar result.
  I had to do this due to the way the halo finding results were initially saved for Phat ELVIS.
  But it should be fairly straight forward to do it.
- I decided to make a separate wrapper function for FIRE.
'''
def infall_criteria_sub_wrapper_function_FIRE(base_dir, sim_num, simnums, BH_parameters, snapnum_info_dict, out_f):
    #
    # Wrapper function for FIRE.
    #
    # Make file names for processed (sub)halo data files.
    processed_dat_fnames = various_halo_file_names_FIRE(base_dir, sim_num)
    #
    print(f"### Identifying infalling subhalos from the disrupted population ###", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    print("# Reading in data #", flush=True, file=out_f)
    # Read in the host halo main branch file.
    t_s_step = time.time()
    host_halo_dict = utilities.read_in_host_main_branch_file_FIRE(processed_dat_fnames, sim_num, BH_parameters)
    print(f"* Host tree main branch data read in:", flush=True, file=out_f)
    #
    # Read in the subtree main branch file: main branch tree file for all subhalos of the host halo.
    subtree_main_branch_df = read_in_tree_main_branch_file_FIRE(processed_dat_fnames, 'subtree')
    print(f"* Subtree main branch data for all subhalos of the host halo read in.", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    #
    # Find infalling subhalos.
    print("# Identifying infalling subhalos #", flush=True, file=out_f)
    t_s_step = time.time()
    infalling_subhalo_from_subtree_dict = find_infalling_subhalos_FIRE(BH_parameters, subtree_main_branch_df, host_halo_dict, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    #
    # Print infalling subhalo identification summary statements.
    summary_statement_FIRE(infalling_subhalo_from_subtree_dict, out_f)
    #
    # Save the result subtrees.
    print("# Saving the infalling subhalo tree result #", flush=True, file=out_f)
    t_s_step = time.time()
    save_infall_subtree_result_FIRE(infalling_subhalo_from_subtree_dict, BH_parameters, sim_num, 'subtree', out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
    #
    print(f"### Identifying infalling subhalos from the surviving population ###", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    print("# Reading in data #", flush=True, file=out_f)
    # Read in the (surviving) tree main branch file: main branch tree file for ALL halos in consistent-trees
    t_s_step = time.time()
    tree_main_branch_df = read_in_tree_main_branch_file_FIRE(processed_dat_fnames, 'tree')
    print(f"* Tree main branch data for all halos read in.", flush=True, file=out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    #
    # Find infalling subhalos.
    print("# Identifying infalling subhalos #", flush=True, file=out_f)
    t_s_step = time.time()
    infalling_subhalo_from_tree_dict = find_infalling_subhalos_FIRE(BH_parameters, tree_main_branch_df, host_halo_dict, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    #
    # Print infalling subhalo identification summary statements.
    summary_statement_FIRE(infalling_subhalo_from_tree_dict, out_f)
    #
    # Save the result trees.
    t_s_step = time.time()
    print("# Saving the infalling subhalo tree result #", flush=True, file=out_f)
    save_infall_subtree_result_FIRE(infalling_subhalo_from_tree_dict, BH_parameters, sim_num, 'tree', out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
#
#
def infall_criteria_sub_wrapper_function(base_dir, sim_num, simnums, BH_parameters, snapnum_info_dict, out_f):
    #
    # Wrapper function for Phat ELVIS.
    #
    # Make file names for processed (sub)halo data files.
    processed_dat_fnames = various_halo_file_names(base_dir, sim_num)
    #
    sim_type = BH_parameters['run_type']
    #
    print("### Reading in data ###", flush=True, file=out_f)
    #
    # Read in the host halo main branch file.
    t_s_step = time.time()
    host_halo_dict = utilities.read_in_host_main_branch_file(processed_dat_fnames, sim_num, sim_type, BH_parameters)
    print(f"* Host tree main branch data read in:", flush=True, file=out_f)  
    #
    # Read in the (disrupted) subtree main branch file: main branch tree file for all subhalos of the host halo
    subtree_main_branch_df = read_in_tree_main_branch_file(processed_dat_fnames, sim_num, sim_type)
    t_e_step = time.time()
    print(f"* Subtree main branch data for all subhalos of the host halo read in.", flush=True, file=out_f)
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    print("", flush=True, file=out_f)
    #
    # Find infalling subhalos.
    t_s_step = time.time()
    print(f"### Identifying infalling subhalos ###", flush=True, file=out_f)
    infalling_subhalo_dict = find_infalling_subhalos(BH_parameters, subtree_main_branch_df, host_halo_dict, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    print("", flush=True, file=out_f)
    #
    '''
    * This part is specific only to the Phat ELVIS project and how its halo catalog and merger tree data are saved:
    '''
    if BH_parameters["two_rockstars"] == 1:
        out_string = (
        "* The original halo finding results from Rockstar on which consistent-trees was run on was not"
        " saved with their particle files. So a separate set of Rockstar results were required just"
        " for the purpose of getting the halo particle data. However, it means that the halo data from"
        " the initial set needs to be matched to the data from the new set."
        )
        print("***** Attention! *****", flush=True, file=out_f)
        print(out_string, flush=True, file=out_f)
        print("", flush=True, file=out_f)
        #
        # For the identified infalling halos, find their matching Rockstar halos from the second Rockstar set.
        print("### Matching subhalos between Rockstar sets for infalling subhalos ###", flush=True, file=out_f)
        t_s_step = time.time()
        infalling_subhalo_dict = connect_rockstar_sets(sim_num, BH_parameters, snapnum_info_dict, infalling_subhalo_dict, out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
        print("", flush=True, file=out_f)
    #
    # Print infalling subhalo identification summary statements.
    summary_statement(infalling_subhalo_dict, out_f)
    #
    # Save the result subtrees.
    t_s_step = time.time()
    print("### Saving the infalling subhalo subtree result ###", flush=True, file=out_f)
    save_infall_subtree_result(infalling_subhalo_dict, BH_parameters, sim_num, sim_type, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
    print("", flush=True, file=out_f)
#
####################################################################################
# Main function
####################################################################################
def main():
    t_s = time.time()
    #
    # Read in Bloodhound parameters.
    BH_parameters = utilities.read_parameters(parameter_fname)
    #
    # Define useful global constants.
    h = BH_parameters['h']
    omega_m = BH_parameters['omega_m']
    temp_cmb = BH_parameters['temp_cmb']
    H0_kpc = 0.1*h # km/s/kpc
    H0_Mpc = 100. * h # km/s/Mpc
    global COSMO, t0
    COSMO = FlatLambdaCDM(H0=H0_Mpc, Om0=omega_m, Tcmb0=temp_cmb)
    t0 = COSMO.age(0).value
    #
    # Open an output statement file.
    text_fdir = BH_parameters["out_statement_dir"]
    text_fname_base = BH_parameters["infall_criteria_out_statement_fname_base"]
    out_f = utilities.open_output_statement_file(text_fdir, text_fname_base)
    # Write the header for the output statement file: description and date/time.
    header_statement = "Selecting subhalos meeting the subhalo selection criteria"
    utilities.write_header_for_result_text_file(header_statement, out_f)
    #
    # Read in snapshot number, redshift, scale factor data.
    snapnum_info_fname = BH_parameters["snapnum_info_fname"]
    sim_name = BH_parameters["simulation_name"]
    snapnum_info_dict = utilities.open_snap_header_file(snapnum_info_fname, sim_name)
    # Add the snapshot time information to the BH parameter dictionary.
    BH_parameters['time_info_dict'] = snapnum_info_dict
    #
    # Get simulation numbers to use, from the parameter dictionary.
    sim_nums = BH_parameters['sim_nums']
    base_dir = BH_parameters['base_dir']
    utilities.print_params(BH_parameters, out_f)
    #
    # Run the infall subhalo finding function.
    for sim_num in sim_nums:
        t_s_step = time.time()
        print("", flush=True, file=out_f)
        print(f"##### Simulation {sim_num} #####", flush=True, file=out_f)
        print("", flush=True, file=out_f)
        # Phat ELVIS:
        if sim_name == 'pELVIS':
            infall_criteria_sub_wrapper_function(base_dir, sim_num, sim_nums, BH_parameters, snapnum_info_dict, out_f)
        elif sim_name == 'FIRE' or 'fire':
            infall_criteria_sub_wrapper_function_FIRE(base_dir, sim_num, sim_nums, BH_parameters, snapnum_info_dict, out_f)
        t_e_step = time.time()
        print(f"##### Simulation {sim_num} finished! #####", flush=True, file=out_f)
        utilities.print_time_taken(t_s_step, t_e_step, "#####", True, out_f)
        print("", flush=True, file=out_f)
    #
    t_e = time.time()
    #
    # Print the total execution time and close the output statement file.
    print("", flush=True, file=out_f)
    print(f"####### Total execution time: {t_e - t_s:.03f} s #######", flush=True, file=out_f)
    #
    out_f.close()
#
if __name__ == "__main__":
    main()