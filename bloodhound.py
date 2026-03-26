#####################################################################################
##### This is the main script file for the subhalo tracking package Bloodhound. #####
##### author: Hyunsu Kong                                                       #####
##### email: hyunsukong@utexas.edu                                              #####
##### website: https://hyunsukong.github.io/                                    #####
##### GitHub (for Bloodhound):                                                  #####
#####################################################################################
'''
Units:

Required input data:
    - 
'''
#-----------------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------------
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
import numpy as np
import pandas as pd
import struct
import sys
import time
#-----------------------------------------------------------------------------------
# Import local modules/libraries/scripts.
#-----------------------------------------------------------------------------------
import halo_analysis
import halo_utilities
import infall_subhalo_criteria
import tree_pre_processing
import utilities
#
#-----------------------------------------------------------------------------------
# Input parameters
#-----------------------------------------------------------------------------------
#parameter_fname = '/scratch/05097/hk9457/FIREII/m12b7e3_sidm1/bloodhound_subhalo_tracking/bloodhound_updating/BH_parameters/bloodhound_parameters.txt'
from config import parameter_fname
#
#-----------------------------------------------------------------------------------
# Fuctions
#-----------------------------------------------------------------------------------
def BH_initialization(parameter_fname, header_statement):
    '''
    * This function handles some of the initialization processes for Bloodhound.
    - Reads in the parameter file.
    - Create all directories Bloodhound needs.
    - Open an output statement file.
    - Write the header for the output statement file: description and date/time.
    - Set two global variables COSMO and T0:
        - COSMO: Astropy FlatLambdaCDM object.
                 Although it is NOT a constant, I follow python's convention and capitalize the variable
                 because, well, it's useful.
        - T0: Age of the universe, constant.
    - Read in snapshot number, redshift, scale factor data.
    - Read in simulation numbers to use, from the parameter dictionary.
    - Returns:
        - BH_parameters: a dictionary containing the parameters read in from the parameter file.
        - sim_nums: a list containing the simulation numbers to use.
        - base_dir: a string containing the path for the parent directory of Bloodhound.
        - out_f: a text file to write output statements.
        - snapnum_info_dict: a dictionary containing the snapshot number, redshift, and scale factor data.
    '''
    #
    # Read in Bloodhound parameters.
    BH_parameters = utilities.read_parameters(parameter_fname)
    #
    # Create various directories.
    utilities.create_directories(BH_parameters)
    #
    # Open an output statement file.
    text_fdir = BH_parameters["out_statement_dir"]
    text_fname_base = BH_parameters["bloodhound_out_statement_fname_base"]
    out_f = utilities.open_output_statement_file(text_fdir, text_fname_base)
    #
    # Write the header for the output statement file: description and date/time.
    utilities.write_header_for_result_text_file(header_statement, out_f)
    # Print the parameters.
    utilities.print_params(BH_parameters, out_f)
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
    #
    # Add a couple of useful items to the parameter data.
    h = BH_parameters['h']
    omega_m = BH_parameters['omega_m']
    temp_cmb = BH_parameters['temp_cmb']
    H0_kpc = 0.1*h # km/s/kpc
    H0_Mpc = 100. * h # km/s/Mpc
    cosmo = FlatLambdaCDM(H0=H0_Mpc, Om0=omega_m, Tcmb0=temp_cmb)
    t0 = cosmo.age(0).value
    BH_parameters['cosmo'] = cosmo
    BH_parameters['t0'] = t0
    #
    # Return the result.
    return(BH_parameters, sim_nums, base_dir, out_f, snapnum_info_dict)
#def halo_particle_tracking_wrapper_function():
'''
* This function
'''
def read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, tree_type, out_f):
    #
    # File name for the infall criteria subtree file.
    fdir = BH_parameters['infall_subtree_out_dir']
    fname_base = BH_parameters['infall_subtree_out_fname_base']
    fname = f"{fdir}/{sim_num}_{tree_type}_{fname_base}.hdf5"
    #
    # Read in the data.
    infall_subtree_df = pd.read_hdf(fname)
    return(infall_subtree_df)
#
def read_in_infalling_subtree_data(BH_parameters, sim_num, out_f):
    #
    # File name for the infall criteria subtree file.
    sim_type = BH_parameters['run_type']
    fdir = BH_parameters['infall_subtree_out_dir']
    fname_base = BH_parameters['infall_subtree_out_fname_base']
    fname = f"{fdir}/{sim_num}_{sim_type}_{fname_base}.hdf5"
    #
    # Read in the data.
    infall_subtree_df = pd.read_hdf(fname)
    return(infall_subtree_df)
'''

'''
def remove_incomplete_subtrees_FIRE(infall_subtree_df, BH_parameters, out_f):
    print(f"* 1) Number of subhalos (both surviving and destroyed/disrupted) in the infalling subtree file: {len(infall_subtree_df.groupby('tree.tid'))}", flush=True, file=out_f)
    # Take only infalling subhalos: infalling? = 1.
    infalling_query = infall_subtree_df.query("`infalling?` == 1")
    print(f"* 2) Number of actually infalling subhalos in 1) (infalling? = 1): {len(infalling_query.groupby('tree.tid'))}", flush=True, file=out_f)
    #
    print("", flush=True, file=out_f)
    return(infalling_query)
#
def remove_incomplete_subtrees(infall_subtree_df, BH_parameters, out_f):
    print(f"* 1) Number of subhalos in the infalling subtree file: {len(infall_subtree_df.groupby('subtree_id'))}", flush=True, file=out_f)
    if BH_parameters["two_rockstars"] == 1:
        # Halo matching between two Rockstar sets was done, so
        # take only subtrees that has a real value for ID.halo.infall: remove those with ID.halo.infall=-1.
        hID_query = infall_subtree_df.query("`ID.halo.infall` != -1")
        #
        # Take only infalling subhalos: infalling? = 1.
        infalling_query = hID_query.query("`infalling?` == 1")
        print(f"* 2) Number of subtrees with a real ID.halo.infall value (not -1): {len(hID_query.groupby('subtree_id'))}", flush=True, file=out_f)
        print(f"* 3) Number of actually infalling subhalos in 2) (infalling? = 1): {len(infalling_query.groupby('subtree_id'))}", flush=True, file=out_f)
    else:
        # Take only infalling subhalos: infalling? = 1.
        infalling_query = infall_subtree_df.query("`infalling?` == 1")
        print(f"* 2) Number of actually infalling subhalos in 1) (infalling? = 1): {len(infalling_query.groupby('subtree_id'))}", flush=True, file=out_f)
    #
    print("", flush=True, file=out_f)
    return(infalling_query)
#
def get_infall_information(infall_subtree_df):
    '''
    * 
    - Every returned array should be ordered by scale.infall from early to late times.
    - For ID.subtree and ID.halo.infall arrays, each element contains an array of IDs corresponding to the given scale.infall.
    '''
    # Initialize the result dictionary.
    infall_information_dict = {}
    sid_list = []
    hid_list = []
    #
    # Get unique arrays for scale.infall and snapshot.infall, preserving the order (uses pandas.series.unique()).
    # Get unique arrays for scale.infall and snapshot.infall, from early to late times (uses numpy.unique()).
    infall_information_dict["scale.factor.infall"] = np.unique(infall_subtree_df['scale.factor.infall'].values)
    infall_information_dict["snapshot.infall"] = np.unique(infall_subtree_df['snapshot.infall'].values)
    #infall_information_dict["scale.infall"] = infall_subtree_df['scale.infall'].unique()
    #infall_information_dict["snapshot.infall"] = infall_subtree_df['snapshot.infall'].unique()
    for i in range(len(infall_information_dict["snapshot.infall"])):
        current_snapnum = infall_information_dict["snapshot.infall"][i]
        current_subtrees = infall_subtree_df.query("`snapshot.infall`== @current_snapnum")
        #
        # Get subtree_id and ID.halo.infall values corresponding to the current scale.infall value.
        sid_list.append(current_subtrees['tree.tid'].unique())
        hid_list.append(current_subtrees['ID.halo.infall'].unique())
    #
    infall_information_dict["ID.subtree"] = sid_list
    infall_information_dict["ID.halo.infall"] = hid_list
    return(infall_information_dict)
#
def get_hIDs_and_num_ps(catalog_ascii_fnames):
    hID_numP_pairs_df_list = []
    for i in range(len(catalog_ascii_fnames)):
        fname = catalog_ascii_fnames[i]
        #
        # Open the .ascii halo catalog file.
        text_df=pd.read_csv(fname, sep=" ", low_memory=False)
        #
        # Take the data part: the first 19 rows contain simulation/snapshot information and I don't need them.
        reduced_data=text_df[19:]
        #
        # Take two columns: id and num_p
        two_columns=reduced_data[['#id', 'num_p']]
        two_columns=two_columns.rename(columns={'#id':'id'})
        two_columns=two_columns.astype(int)
        #
        # Append the dataframe to the result list.
        hID_numP_pairs_df_list.append(two_columns)
    return(hID_numP_pairs_df_list)
#
def get_rockstar_particle_ID_data(catalog_bin_fnames, hID_numP_pairs_df_list, out_f):
    particle_ID_list = []
    for i in range(len(catalog_bin_fnames)):
        # File name for the .bin Rockstar result file.
        bin_fname = catalog_bin_fnames[i]
        #
        # Number of halos in the current catalog block.
        hID_numP_pairs_df = hID_numP_pairs_df_list[i]
        num_halos=len(hID_numP_pairs_df)
        #
        # Get the number of particles array for all halos in the current file.
        num_p_arr = hID_numP_pairs_df.num_p.values
        #
        # Open the binary file.
        bin_file = open(bin_fname, 'rb')
        #
        # Skip header and halo tables.
        bin_file.seek(256+296*num_halos)
        #
        # Get integer particle IDs for each halo and attend the particle ID arrays (tuples) to a list.
        pID_list_current_block = []
        for num_p in num_p_arr:
            particle_IDs = struct.unpack("Q" * num_p, bin_file.read(8 * num_p))
            pID_list_current_block.append(particle_IDs)
        particle_ID_list.append(pID_list_current_block)
        #
        # Check if EOF has been reached: "not remaining_dat" is True if remaining_dat is empty.
        remaining_dat = bin_file.read()
        #if not remaining_dat:
        #    print(f"  * Block {i}: successfully reached EOF!", flush=True, file=out_f)
        if remaining_dat:
            print(f"  * Block {i}: EOF not reached!", flush=True, file=out_f)
            print(f"    * Remaining data is: {remaining_dat}", flush=True, file=out_f)
        #
        # Close the current binary file.
        bin_file.close()
    #
    # Return the particle ID list.
    return(particle_ID_list)
#
def check_num_halos(particle_ID_list, hID_numP_pairs_df_list, out_f):
    if len(particle_ID_list) != len(hID_numP_pairs_df_list):
        print("  * Number of blocks for particle_ID_list and hID_numP_pairs_df_list are different!", flush=True, file=out_f)
    for i in range(len(particle_ID_list)):
        if len(particle_ID_list[i]) != len(hID_numP_pairs_df_list[i]):
            print(f"  * Block {i}: number of halos in particle ID data and hID-numP pair data are different!", flush=True, file=out_f)
#
def get_particle_IDs_of_halo(hID, hID_numP_pairs_df_list, particle_ID_list, out_f):
    '''
    
    - hID_numP_pairs_df_list and particle_ID_list have exactly the same structure.
    '''
    for i in range(len(hID_numP_pairs_df_list)):
        current_block_hID_arr = hID_numP_pairs_df_list[i].id.values
        #
        # Check if the current block contains the current hID.
        if hID in current_block_hID_arr:
            block_idx = i
            halo_idx = np.where(current_block_hID_arr==hID)[0][0]
            break
        else:
            continue
    #
    # Get the particle IDs for the current halo: the result is a tuple, so convert it to an array.
    pID_arr = np.array(particle_ID_list[block_idx][halo_idx])
    #
    # Test if the number of particles in pID_arr is the same as that given by the num_p column in hID_numP_pairs_df_list.
    # This should always be true, but it might be useful to have a error message when it isn't!
    if hID_numP_pairs_df_list[block_idx].num_p.values[halo_idx] != len(pID_arr):
        print(f"  * Number of particles inconsistent! halo ID: {hID}, block index: {block_idx}, halo index: {halo_idx}", flush=True, file=out_f)
    #
    # Return the particle IDs.
    return(pID_arr)
#
def get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f):
    '''
    '''
    # Particle IDs for each halo will be appended to a list in infall_information_dict.
    infall_information_dict['ID.particle'] = []
    #
    # Number of rockstar output files per snapshot.
    num_rockstar_files = BH_parameters['num_rockstar_files']
    #
    #
    if BH_parameters['simulation_name']=='pELVIS':
        halo_finding_output_dir = f"{BH_parameters['base_dir']}/{BH_parameters['run_type']}/halo_{sim_num}/rockstar_catalogs/rockstar_output"
    elif BH_parameters['simulation_name']=='FIRE':
        halo_finding_output_dir = f"{BH_parameters['base_dir']}/halo/rockstar_dm/catalog"
    #
    #
    infall_snapshot_arr = infall_information_dict["snapshot.infall"]
    infall_hid_list = infall_information_dict["ID.halo.infall"]
    ####### test #######
    #infall_snapshot_arr = infall_snapshot_arr[72:85]
    #infall_hid_list = infall_hid_list[72:85]
    #
    for i in range(len(infall_snapshot_arr)):
        current_snap = infall_snapshot_arr[i]
        current_snap_infall_hid_arr = infall_hid_list[i]
        #
        # Read in Rockstar halo particle data for the current snapshot.
        catalog_ascii_fnames = utilities.make_rockstar_fnames(halo_finding_output_dir, num_rockstar_files, current_snap, 'ascii')
        catalog_bin_fnames = utilities.make_rockstar_fnames(halo_finding_output_dir, num_rockstar_files, current_snap, 'bin')
        hID_numP_pairs_df_list = get_hIDs_and_num_ps(catalog_ascii_fnames)
        #
        print(f"* Reading in halo particle ID data for snapshot {current_snap}... ", end="", flush=True, file=out_f)
        t_s_step = time.time()
        particle_ID_list = get_rockstar_particle_ID_data(catalog_bin_fnames, hID_numP_pairs_df_list, out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
        #
        # Just in case, check that the number of Rockstar blocks and number of halos are consistent between
        # the data read in from the .bin file and the .ascii file.
        check_num_halos(particle_ID_list, hID_numP_pairs_df_list, out_f)
        #
        # Get particle IDs of halos infalling at the current snapshot.
        current_snap_pID_list = []
        for j in range(len(current_snap_infall_hid_arr)):
            current_hID = current_snap_infall_hid_arr[j]
            current_halo_pID_tuple = get_particle_IDs_of_halo(current_hID, hID_numP_pairs_df_list, particle_ID_list, out_f)
            current_snap_pID_list.append(current_halo_pID_tuple)
        #
        # Append the pID list for the current snapshot to the result dictionary.
        infall_information_dict['ID.particle'].append(current_snap_pID_list)
    #
    # Convert the final infall particle ID list to an array.
    infall_information_dict['ID.particle'] = np.array(infall_information_dict['ID.particle'], dtype=object)
    #
    return(infall_information_dict)
#
def initialize_halo_tracking_FIRE(BH_parameters, sim_num, out_f):
    '''
    * This function initializes subhalo tracking by retrieving the particle ID data for identified infalling subhalos.
    '''
    # Read in the infalling subtree data for the given simulation number.
    print("# Reading in infalling subhalo data identified by infalling_subhalo_criteria.py #", flush=True, file=out_f)
    infall_subtree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'subtree', out_f)
    infall_tree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'tree', out_f)
    # Merge the two dataframes.
    infall_subhalo_df = pd.concat([infall_subtree_df, infall_tree_df])
    #
    # Remove broken-link subhalos.
    cleaned_infall_subtree_df = remove_incomplete_subtrees_FIRE(infall_subhalo_df, BH_parameters, out_f)
    #
    # Get ID.subtree and ID.halo.infall arrays organized according to the infall time.
    infall_information_dict = get_infall_information(cleaned_infall_subtree_df)
    #
    t_s_step = time.time()
    print("# Getting the halo particle data for infalling subhalos #", flush=True, file=out_f)
    infall_information_dict = get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
    #
    # Return the infalling subhalo data dictionary.
    ##### Test #####
    #test_print_dict = {"snapshot.infall":infall_information_dict["snapshot.infall"],
    #                  "ID.halo.infall":infall_information_dict["ID.halo.infall"]}
    #pd.set_option('display.max_rows', None)
    #print(pd.DataFrame.from_dict(test_print_dict, orient='columns'))
    return(infall_information_dict)
#
def initialize_halo_tracking(BH_parameters, sim_num, out_f):
    '''
    * This function initializes subhalo tracking by retrieving the particle ID data for identified infalling subhalos.
    '''
    # Read in the infalling subtree data for the given simulation number.
    print("# Reading in infalling subhalo data identified by infalling_subhalo_criteria.py #", flush=True, file=out_f)
    infall_subtree_df = read_in_infalling_subtree_data(BH_parameters, sim_num, out_f)
    #
    # Remove broken-link subhalos (and unmatched halos if halo matching between two Rockstar sets was done).
    cleaned_infall_subtree_df = remove_incomplete_subtrees(infall_subtree_df, BH_parameters, out_f)
    #
    # Get ID.subtree and ID.halo.infall arrays organized according to the infall time.
    infall_information_dict = get_infall_information(cleaned_infall_subtree_df)
    #
    t_s_step = time.time()
    print("# Getting the halo particle data for infalling subhalos #", flush=True, file=out_f)
    infall_information_dict = get_infall_particle_IDs(infall_information_dict, BH_parameters, sim_num, out_f)
    t_e_step = time.time()
    utilities.print_time_taken(t_s_step, t_e_step, "#", True, out_f)
    print("", flush=True, file=out_f)
    #
    # Return the infalling subhalo data dictionary.
    return(infall_information_dict)
#
def initialize_snapshot_data_FIRE(BH_parameters, sim_num, snap_num, base_dir, use_argsort, out_f):
    '''
    * This function uses the SnapshotData class from utilities.py to read in the snapshot data.
    '''
    # Number of output blocks per snapshot.
    blocks = BH_parameters['num_output_files']
    snapshot_data_dict = {}
    # Initialize the snapshot dictionary-class.
    snapshot_data_dict['snapshot_data'] = utilities.SnapshotData_FIRE(sim_num, snap_num, base_dir, blocks)
    #
    for i in range(len(snapshot_data_dict)):
        #
        # Read in the simulation snapshot output data.
        print(f"* Reading in the simulation output data: snapshot {snap_num} ... ", end="", flush=True, file=out_f)
        t_s_step = time.time()
        snapshot_data_dict['snapshot_data'].read_in_snapshot_data(use_argsort)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
        #
        # Read in the particle ID sort index data, if it exists.
        # I don't think use_argsort=True will ever happen.
        # Now, I am sorting the particle ID data anyway!
        if use_argsort:
            print(f"* Reading in the particle ID sort index data: snapshot {snap_num} ... ", end="", flush=True, file=out_f)
            t_s_step = time.time()
            snapshot_data_dict['snapshot_data'].read_in_pID_argsort_data()
            t_e_step = time.time()
            utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
    #
    return(snapshot_data_dict)
#
def initialize_snapshot_data(sim_num, snap_num, base_dir, sim_types, use_argsort, out_f):
    '''
    * This function uses the SnapshotData class from utilities.py to read in the snapshot data.
    - Need to add a script that checks if the snapshot number is below 38.
      - For <38, snap_disk = snap_dmo!
    '''
    snapshot_data_dict = {}
    #
    # For Phat ELVIS, Disk simulations are identical to DMO simulations for snapshots below 38.
    # So if snap_num < 38, set 'disk' to 'dmo'.
    # I am not sure if this is the best way as this would read in the same DMO data twice, but let's go with it for now.
    if snap_num < 38:
        for i in range(len(sim_types)):
            sim_types[i] = 'dmo'
    #
    if len(sim_types) == 1:
        if sim_types[0] == 'disk':
            blocks=1
        elif sim_types[0] == 'dmo':
            if sim_num == 493:
                blocks = 1
            else:
                blocks = 8
        #
        # Initialize the snapshot dictionary-class.
        snapshot_data_dict[sim_types[0]] = utilities.SnapshotData(sim_num, snap_num, base_dir, sim_types[0], blocks)
    else:
        for i in range(len(sim_types)):
            if sim_types[i] == 'disk':
                blocks=1
            elif sim_types[i] == 'dmo':
                if sim_num == 493:
                    blocks = 1
                else:
                    blocks = 8
            #
            # Initialize the snapshot dictionary-class.
            snapshot_data_dict[sim_types[i]] = utilities.SnapshotData(sim_num, snap_num, base_dir, sim_types[i], blocks)
    #
    for i in range(len(snapshot_data_dict)):
        #
        # Read in the simulation snapshot output data.
        print(f"* Reading in the simulation output data: snapshot {snap_num}, {sim_types[i]}... ", end="", flush=True, file=out_f)
        t_s_step = time.time()
        snapshot_data_dict[sim_types[i]].read_in_snapshot_data(use_argsort)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
        #
        # Read in the particle ID sort index data, if it exists.
        if use_argsort:
            print(f"* Reading in the particle ID sort index data: snapshot {snap_num}, {sim_types[i]}... ", end="", flush=True, file=out_f)
            t_s_step = time.time()
            snapshot_data_dict[sim_types[i]].read_in_pID_argsort_data()
            t_e_step = time.time()
            utilities.print_time_taken(t_s_step, t_e_step, "*" ,True, out_f)
    #
    return(snapshot_data_dict)
#
def track_particles(snapshot_obj, halo_pID, use_argsort):
    '''
    * This function finds particles in the given halo_pID data from the given snapshot_obj data.
    '''
    if use_argsort==True:
        # If the particle ID argsort data exists, use it to retrieve snapshot's particles IDs that correspond to the given halo.
        # It's much faster to use it.
        pID_idx_in_snap = snapshot_obj["pID.sort_idx"][halo_pID]
    else:
        # If the particle ID argsort data doesn't exist, look for the particle IDs using numpy.isin()
        #pID_idx_in_snap = np.nonzero(np.isin(snapshot_obj["ID.particle"], halo_pID))[0]
        # *** The new version does arg_sort when reading in the particle data, so use it!!
        # Shift the halo particle IDs by the smallest particle ID in the entire simulation: then halo particle IDs can be used as indices.
        halo_pID_shifted = halo_pID - snapshot_obj['ID.particle.min']
        pID_idx_in_snap = snapshot_obj["pID.sort_idx"][halo_pID_shifted]
    #
    # Get coordinates and velocities for the particles.
    tracked_coords = snapshot_obj["Coordinates"][pID_idx_in_snap]
    tracked_vels = snapshot_obj["Velocities"][pID_idx_in_snap]
    #
    # Return results as a list.
    #print(len(halo_pID), len(tracked_coords))
    return([tracked_coords, tracked_vels])
#
def remove_odd_pIDs(hID, snapshot_obj, halo_pID, use_argsort, out_f):
    '''
    *** Outdated ***
    * For some odd reason, there are very rare cases where pID_arr contains IDs that are 
      greater than the total number of particles in the snapshot.
      Check it and remove those pIDs from pID_arr.
      Also print their halo ID!
    * The "odd" particle ID cases happens because GIZMO's particle IDs are numbered continuously from PartType0 to PartTypeN and Rockstar uses PartType1 (DM) and PartType2 (low-res DM).
    * So it's actually not "odd" and this function should not be used.
    '''
    if use_argsort==True:
        snapshot_num_particles = len(snapshot_obj["pID.sort_idx"])
    else:
        snapshot_num_particles = len(snapshot_obj["ID.particle"])
    #
    pID_check_mask = np.where(halo_pID > snapshot_num_particles)[0]
    if len(pID_check_mask) > 0:
        print(f"  * Attention! Halo {hID}: particle IDs from Rockstar contains IDs that are greater then the total number of particles in the snapshot {snapshot_num_particles}", flush=True, file=out_f)
        print(f"  * These particles will be removed from the analysis!", flush=True, file=out_f)
        print(f"  * Number of halo particles from Rockstar: {len(halo_pID)}", flush=True, file=out_f)
        print(f'  * Number of "BAD" particles: {len(pID_check_mask)}', flush=True, file=out_f)
        halo_pID = np.delete(halo_pID, pID_check_mask)
    return(halo_pID)
#
def make_subhalo_catalog(BH_parameters, sim_num, full_tracking_df_list, out_f):
    '''
    * This function takes in the full tracking data for all tracked subhalos (list of dataframes), and constructs a subhalo catalog.
      - Properties such as x, y, z, vmax, etc. are taken at the last snapshot of the subhalo:
        - surviving: last snapshot of the simulation
        - destroyed: at the snapshot of disruption
      - It retrieves the tree.tid information from the infall tree/subtree data,
      - Input:
        - full_tracking_df_list - list of dataframes where each dataframe is a tracking data of a subhalo.
      - Data: all values are in non-h units.
        - ID.halo.infall:
        - ID.tree: tree ID of the subhalo in the merger tree data, taken from the last snapshot (old name: tree.tid)
        - snapshot.infall:
        - scale.infall:
        - number.of.particles.infall: number of particles that were assigned to the halo at the infall snapshot
        - scale.disrupt:
        - rmax.infall:
        - rmax.peak:
        - rmax:
        - x, y, z: comoving kpc
        - host.x, host.y, host.z: physical kpc
        - distance.from.host.ckpc: comoving kpc
        - closest.pericenter: physical kpc
    '''
    # Initialize the result dictionary.
    result_dict = {
        "ID.tree":[],
        "ID.halo.infall":[],
        "snapshot.infall":[],
        "scale.factor.infall":[],
        "number.of.particles.infall":[],
        "is.infalling":[],
        "scale.factor.disrupt":[],
        "vmax":[],
        "vmax.infall":[],
        "vmax.peak":[],
        "scale.factor.vmax.peak":[],
        "rmax":[],
        "rmax.infall":[],
        "rmax.peak":[],
        "scale.radius.klypin":[],
        "radius.boundary":[],
        "closest.pericenter":[],
        "scale.factor.closest.pericenter":[],
        "number.of.pericenters":[],
        "distance.from.host.ckpc":[],
        "x":[],
        "y":[],
        "z":[],
        "host.x":[],
        "host.y":[],
        "host.z":[],
        "vx":[],
        "vy":[],
        "vz":[],
        "host.vx":[],
        "host.vy":[],
        "host.vz":[],
        "host.vrad":[],
        "time.survival.gyr":[]
    }
    # Read in the infall tree and subtree data to get the tree.tid information.
    #if BH_parameters['simulation_name'] == 'pELVIS':
    #    infalling_subtree_df = read_in_infalling_subtree_data(BH_parameters, sim_num, out_f)
    if BH_parameters['simulation_name'] == 'FIRE':
        infall_subtree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'subtree', out_f)
        infall_tree_df = read_in_infalling_subtree_data_FIRE(BH_parameters, sim_num, 'tree', out_f)
        # Merge the two dataframes.
        infall_subhalo_tree_df = pd.concat([infall_subtree_df, infall_tree_df])
    for i in range(len(full_tracking_df_list)):
        current_df = full_tracking_df_list[i]
        # Some information (e.g., ID.halo.infall, snapshot.infall, particle.mass etc.) can be read in from any row. Get it from the first row.
        first_row = current_df.iloc[0] # The first row read in as pandas series.
        ID_halo = int(first_row['ID.halo.infall'])
        snap_infall = int(first_row['snapshot.infall'])
        scale_disrupt = first_row['scale.factor.disrupt']
        num_particles_infall = int(first_row['number.of.particles.infall'])
        # Remove data after disruption: if the subhalo doesn't disrupt, end_idx is simply that of the last snapshot.
        if scale_disrupt == -1:
            # subhalo doesn't disrupt.
            end_idx = None
        else:
            # subhalo disrupts: +1 to include the disruption snapshot.
            end_idx = np.where(np.isclose(current_df['scale.factor'], scale_disrupt, atol=1e-4))[0][0] + 1
        # Slice the tracking data from infall to the last (z=0 or disruption) snapshot.
        current_df = current_df[:end_idx]
        # Identify the current subhalo in the tree data to get the tree.tid data:
        #  - for most cases, queryig by halo_ID should be enough. But also use infall_snap just in case there are more than one subhalo with the same infall halo ID.
        current_halo_tree_df = infall_subhalo_tree_df.query("`ID.halo.infall`==@ID_halo and `snapshot.infall`==@snap_infall")
        tree_first_row = current_halo_tree_df.iloc[0]
        ID_tree = int(tree_first_row['tree.tid'])
        scale_infall = tree_first_row['scale.factor.infall']
        is_it_infalling = int(tree_first_row['infalling?'])
        # Get the halo properties at the infall, peak, and "last" (either at z=0 or disruption) snapshots.
        vmax_arr = current_df['vmax'].values
        rmax_arr = current_df['rmax'].values
        scale_arr = current_df['scale.factor'].values
        vmax_last = vmax_arr[-1]
        vmax_infall = vmax_arr[0]
        vmax_peak_idx = np.argmax(vmax_arr)
        vmax_peak = vmax_arr[vmax_peak_idx]
        rmax_last = rmax_arr[-1]
        rmax_infall = rmax_arr[0]
        rmax_peak = rmax_arr[vmax_peak_idx]
        scale_vmax_peak = scale_arr[vmax_peak_idx]
        scale_last = scale_arr[-1] # Either 1 (surviving) or the disruption time.
        last_row = current_df.iloc[-1]
        x = last_row.x
        y = last_row.y
        z = last_row.z
        rel_x = last_row['host.x']
        rel_y = last_row['host.y']
        rel_z = last_row['host.z']
        distance = last_row['distance.from.host.ckpc']
        vx = last_row.vx
        vy = last_row.vy
        vz = last_row.vz
        rel_vx = last_row['host.vx']
        rel_vy = last_row['host.vy']
        rel_vz = last_row['host.vz']
        rel_vrad = last_row['host.velocity.rad']
        rs_klypin = last_row['scale.radius.klypin']
        t_lb_infall = BH_parameters['cosmo'].lookback_time(1./scale_infall - 1.).value
        t_lb_disrupt = BH_parameters['cosmo'].lookback_time(1./scale_last - 1.).value
        t_surv = t_lb_infall - t_lb_disrupt
        r_boundary = last_row['radius.boundary']
        # Append result to the dictionary.
        result_dict['ID.tree'].append(ID_tree)
        result_dict['ID.halo.infall'].append(ID_halo)
        result_dict['snapshot.infall'].append(snap_infall)
        result_dict['scale.factor.infall'].append(scale_infall)
        result_dict['number.of.particles.infall'].append(num_particles_infall)
        result_dict['is.infalling'].append(is_it_infalling)
        result_dict['scale.factor.disrupt'].append(scale_last)
        result_dict['vmax'].append(vmax_last)
        result_dict['vmax.infall'].append(vmax_infall)
        result_dict['vmax.peak'].append(vmax_peak)
        result_dict['scale.factor.vmax.peak'].append(scale_vmax_peak)
        result_dict['rmax'].append(rmax_last)
        result_dict['rmax.infall'].append(rmax_infall)
        result_dict['rmax.peak'].append(rmax_peak)
        result_dict['scale.radius.klypin'].append(rs_klypin)
        result_dict['radius.boundary'].append(r_boundary)
        result_dict['closest.pericenter'].append(first_row['closest.pericenter'])
        result_dict['scale.factor.closest.pericenter'].append(first_row['scale.factor.closest.pericenter'])
        result_dict['number.of.pericenters'].append(int(first_row['number.of.pericenters']))
        result_dict['distance.from.host.ckpc'].append(distance)
        result_dict['x'].append(x)
        result_dict['y'].append(y)
        result_dict['z'].append(z)
        result_dict['host.x'].append(rel_x)
        result_dict['host.y'].append(rel_y)
        result_dict['host.z'].append(rel_z)
        result_dict['vx'].append(vx)
        result_dict['vy'].append(vy)
        result_dict['vz'].append(vz)
        result_dict['host.vx'].append(rel_vx)
        result_dict['host.vy'].append(rel_vy)
        result_dict['host.vz'].append(rel_vz)
        result_dict['host.vrad'].append(rel_vrad)
        result_dict['time.survival.gyr'].append(t_surv)
    # Convert the subhalo catalog dictionary to a pandas dataframe
    result_df = pd.DataFrame.from_dict(result_dict, orient='columns')
    return(result_df)       
#
def subhalo_tracking_wrapper_function(BH_parameters, sim_num, infall_information_dict, out_f):
    '''
    * This is a wrapper function for performing subhalo particle tracking for all subhalos.
    '''
    use_argsort = BH_parameters['pID_argsort_made'] # 0: false, 1: true, making it a variable for readability.
    #
    # Create an array of snapshot numbers to use.
    # Think about forward tracking and backward tracking: skip this for now.
    last_snapnum = BH_parameters['last_snapnum']
    infall_snapnums = infall_information_dict["snapshot.infall"]
    first_infalling_snap = np.min(infall_snapnums)
    last_infalling_snap = np.max(infall_snapnums)
    track_snapnums = np.arange(np.min(infall_snapnums), last_snapnum+1, 1)
    print("* Infall snapshot numbers:", flush=True, file=out_f)
    print(f"  {infall_snapnums}", flush=True, file=out_f)
    print("* Snapshot numbers to use for tracking:", flush=True, file=out_f)
    print(f"  {track_snapnums}", flush=True, file=out_f)
    print("", flush=True, file=out_f)
    #
    infall_hid_arr = infall_information_dict["ID.halo.infall"]
    infall_halo_pID_arr = infall_information_dict['ID.particle']
    #
    # Perform halo tracking at each snapshot: retrieves the coordinates and velocities of halo particles at each snapshot.
    num_subs_tracking_started = 0
    num_subs_last_infall_snap = 0
    ##### Test #####
    #infall_snapnums = infall_snapnums[72:85]
    #track_snapnums = np.arange(np.min(infall_snapnums), 150, 1)
    ##### Test #####
    for i in range(len(track_snapnums)):
        t_s_snap = time.time()
        current_snap = track_snapnums[i]
        ######### Test ##########
        #if current_snap < 144:
        #    continue
        ######### Test ##########
        print(f"# Current snapshot: {current_snap} #", flush=True, file=out_f)
        #
        # Initialize the snapshot dictionary which contains a snapshot dictionary-class for each element in BH_parameters['tracking_order'].
        if BH_parameters['simulation_name'] == 'pELVIS':
            snapshot_data_dict = initialize_snapshot_data(sim_num, current_snap, BH_parameters['base_dir'], BH_parameters['tracking_order'], use_argsort, out_f)
        elif BH_parameters['simulation_name'] == 'FIRE':
            snapshot_data_dict = initialize_snapshot_data_FIRE(BH_parameters, sim_num, current_snap, BH_parameters['base_dir'], use_argsort, out_f)
        #
        # For the first snapshot used, print some useful information.
        #if i >= 0:
        if i == 0:
            print("* Snapshot file information:", flush=True, file=out_f)
            print("  * Only for the first snapshot used", flush=True, file=out_f)
            print("  * Only for the first block, if multiple blocks", flush=True, file=out_f)
            for key in snapshot_data_dict:
                snapshot_data = snapshot_data_dict[key]
                print(f"  * File path/name: {snapshot_data['file.path']}", flush=True, file=out_f)
                snapshot_data.print_header(out_f)
                print("", flush=True, file=out_f)
        #
        if current_snap == last_infalling_snap:
            print(f"*** Snapshot {current_snap} is the last snapshot with infalling subhalos! ***", flush=True, file=out_f)
        #
        if current_snap in infall_snapnums:
            # There are infalling subhalos in the current snapshot.
            infall_idx = np.where(infall_snapnums == current_snap)[0][0]
            infall_snap = infall_snapnums[infall_idx]
            current_snap_infalling_hids = infall_hid_arr[infall_idx]
            current_snap_infalling_halos_pID_arr = infall_halo_pID_arr[infall_idx]
            #
            # Update the number of subhalos that have already started tracking.
            num_subs_tracking_started += num_subs_last_infall_snap
            num_subs_last_infall_snap = len(current_snap_infalling_hids)
            #
            # Infall snapshot numbers that have been used already, excluding current_snap.
            infall_snapnums_used = infall_snapnums[:infall_idx]
            #
            # Tracking halos infalling at the current snapshot.
            print(f"  * Tracking {len(current_snap_infalling_hids)} subhalo(s) infalling at the current snapshot ({infall_snap})...", flush=True, file=out_f)
            t_s_step = time.time()
            for j in range(len(current_snap_infalling_hids[:])):
                # Get halo ID and particle IDs for the current halo.
                current_hID = current_snap_infalling_hids[j]
                current_halo_pIDs = current_snap_infalling_halos_pID_arr[j]
                #if current_hID == 18490:
                #    print()
                #    print("18490 located!!")
                #    print()
                #print(current_halo_pIDs)
                #print(np.max(current_halo_pIDs))
                #
                # Tracking halo particles.
                for key in snapshot_data_dict:
                    # Remove particle IDs that are larger than the total number of particles in the snapshot.
                    # This is weird, but they exist for a very rare number of cases.
                    #current_halo_pIDs = remove_odd_pIDs(current_hID, snapshot_data_dict[key], current_halo_pIDs, use_argsort, out_f)
                    infall_information_dict['ID.particle'][infall_idx][j] = current_halo_pIDs
                    # Get halo particles' coordinates and velocities at the current snapshot.
                    tracked_particle_data = track_particles(snapshot_data_dict[key], current_halo_pIDs, use_argsort)
                    #
                    # Save tracked particle data in a .hdf5 file.
                    utilities.output_halo_particles_hdf5(BH_parameters, sim_num, current_hID, infall_snap, [current_snap], current_halo_pIDs, [tracked_particle_data], key, out_f)
            #
            t_e_step = time.time()
            utilities.print_time_taken(t_s_step, t_e_step, "  *", True, out_f)
        #
        else:
            # There are no infalling subhalos in the current snapshot.
            print("  * There are no infalling subhalos at the current snapshot!", flush=True, file=out_f)
            #
            # Infall snapshot numbers that have been used already, excluding current_snap.
            infall_snapnums_used = infall_snapnums[:infall_idx+1]
            #
            # Update the number of subhalos that have already started tracking.
            num_subs_tracking_started += num_subs_last_infall_snap
            # If there are consecutive snapshots with no infalling subhalos, we do not want the count
            # of num_subs_tracking_started to increase after the first one.
            num_subs_last_infall_snap = 0
        #
        # Track subhalos that infalled already and started their tracking at an earlier snapshot (i.e. infall_snapnums_used).
        t_s_step = time.time()
        print(f"  * Tracking {num_subs_tracking_started} subhalos that started tracking at earlier snapshots: t_infall < t_current... ", flush=True, file=out_f)
        print(f"    * {infall_snapnums_used}", flush=True, file=out_f)
        for j in range(len(infall_snapnums_used)):
            earlier_infall_snap = infall_snapnums_used[j]
            earlier_snap_infalling_hids = infall_hid_arr[j]
            earlier_snap_infalling_halos_pID_arr = infall_halo_pID_arr[j]
            for k in range(len(earlier_snap_infalling_hids)):
                # Get halo ID and particle IDs for the current halo.
                current_hID = earlier_snap_infalling_hids[k]
                current_halo_pIDs = earlier_snap_infalling_halos_pID_arr[k]
                #print(np.max(current_halo_pIDs))
                # Tracking halo particles.
                for key in snapshot_data_dict:
                    # Get halo particles' coordinates and velocities at the current snapshot.
                    tracked_particle_data = track_particles(snapshot_data_dict[key], current_halo_pIDs, use_argsort)
                    #
                    # Save tracked particle data in a .hdf5 file.
                    utilities.output_halo_particles_hdf5(BH_parameters, sim_num, current_hID, earlier_infall_snap, [current_snap], current_halo_pIDs, [tracked_particle_data], key, out_f)
        #
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "  *", True, out_f)
        #
        t_e_snap = time.time()
        utilities.print_time_taken(t_s_snap, t_e_snap, "#", True, out_f)
        print("", flush=True, file=out_f)
#
def subhalo_analysis_wrapper_function(BH_parameters, sim_num, snapnum_info_dict, out_f):
    '''
    * This is a wrapper function for performing subhalo analysis for all subhalos tracked in previous steps of Bloodhound.
    - Need file paths for
      - host tree main branch file: DMO and Disk
      - DMO surviving halo main branch file
      - DMO subtree main branch file
      - Disk subtree file with subhalos found by using the infall criteria
    '''
    t_s_step = time.time()
    tracked_halo_particle_dir = BH_parameters["tracked_halo_particle_dir"]
    # Empty list to append the (tree-like) halo tracking data to.
    full_tracking_df_list = []
    if BH_parameters['simulation_name'] == 'pELVIS':
        # Make the file names for host halo main tree data: host_halo_dat_fnames will also contain file names for other data, but these will not be used.
        host_halo_dat_fnames = utilities.various_halo_file_names(BH_parameters['base_dir'], sim_num)
        # Directory path for tracked halo particles.
        dpath = f"{tracked_halo_particle_dir}/{sim_type}/{sim_num}"
        print(dpath, flush=True, file=out_f)
        sim_types = BH_parameters["tracking_order"]
        for i in range(len(sim_types)):
            sim_type = sim_types[i]
            # Read in the host halo main branch data.
            # Everything is read in with non-h units.
            host_halo_dict = utilities.read_in_host_main_branch_file(host_halo_dat_fnames, sim_num, sim_type, BH_parameters)
            #print()
            #print(f"sim_type: {sim_type}")
            # Directory path to all tracked halo particle data for the current simulation number and type
            dpath = f"{tracked_halo_particle_dir}/{sim_type}/{sim_num}"
            print(dpath, flush=True, file=out_f)
            hID_arr, infall_snap_arr, full_name_arr = utilities.get_halo_particle_file_names_in_dir(dpath)
            print(f"{sim_type}, {len(hID_arr)}, {len(infall_snap_arr)}, {len(full_name_arr)}", flush=True, file=out_f)
            peri_parameters = tree_pre_processing.peri_parameter_dict(BH_parameters['target_t_space_gyr'], BH_parameters['target_savgol_window_gyr'], BH_parameters['vrad_sign_frac_threshold'], host_halo_dict['scale.factor'], BH_parameters, out_f)
            # Merge the pericenter parameter dictionary to BH_parameters.
            BH_parameters = BH_parameters | peri_parameters
            for j in range(len(hID_arr)):
                current_hID = hID_arr[j]
                current_infall_snap = infall_snap_arr[j]
                current_fname = full_name_arr[j]
                current_halo = halo_analysis.analyze_halo(current_hID, current_infall_snap, current_fname, host_halo_dict, snapnum_info_dict, BH_parameters, out_f)
                # Append the tracking dataframe to the result list.
                full_tracking_df_list.append(current_halo['tracking.df'])
                # Still need to finish this section: most of it will be similar to the FIRE version.
    elif BH_parameters['simulation_name'] == 'FIRE':
        # Make the file names for host halo main tree data: host_halo_dat_fnames will also contain file names for other data, but these will not be used.
        host_halo_dat_fnames = utilities.various_halo_file_names_FIRE(BH_parameters['base_dir'], sim_num)
    # Directory path for tracked halo particles.
        # Read in the host halo main branch data.
        # Everything is read in with non-h units.
        host_halo_dict = utilities.read_in_host_main_branch_file_FIRE(host_halo_dat_fnames, sim_num, BH_parameters)
        # Directory path for tracked halo particles.
        dpath = tracked_halo_particle_dir
        # Get tracked halo particle file information in the directory.
        hID_arr, infall_snap_arr, full_name_arr = utilities.get_halo_particle_file_names_in_dir(dpath)
        print(f"* Number of tracked subhalo particle files to read in: {len(hID_arr)}", flush=True, file=out_f)
        # Parameters to use for the pericenter computation
        peri_parameters = tree_pre_processing.peri_parameter_dict(BH_parameters['target_t_space_gyr'], BH_parameters['target_savgol_window_gyr'], BH_parameters['vrad_sign_frac_threshold'], host_halo_dict['scale.factor'], BH_parameters, out_f)
        # Merge the pericenter parameter dictionary to BH_parameters.
        BH_parameters = BH_parameters | peri_parameters
        # Compute the distance interpolation factor: do this once here rather than once for each subhalo.
        # Set the minimum at 2.
        #interp_scale_target = 0.001 # Desired time step size for interpolation, in terms of scale factor.
        #median_scale_spacing = np.median(host_halo_dict['scale'][1:] - host_halo_dict['scale'][:-1])
        #interp_factor = int(round(median_scale_spacing / interp_scale_target))
        #if interp_factor < 2:
        #    interp_factor = 2
        #BH_parameters['interpolation_factor'] = interp_factor
        #
        # Perform halo analysis on each subhalo.
        for j in range(len(hID_arr[:])):
            t_s_halo = time.time()
            current_hID = hID_arr[j]
            current_infall_snap = infall_snap_arr[j]
            current_fname = full_name_arr[j]
            print(f"    * Analyzing... Halo ID-{current_hID}, Infall snapshot-{current_infall_snap} ", end="", flush=True, file=out_f)
            # Perform subhalo analysis for the current subhalo.
            current_halo = halo_analysis.analyze_halo(current_hID, current_infall_snap, current_fname, host_halo_dict, snapnum_info_dict, BH_parameters, out_f)
            # Append the tracking dataframe to the result list.
            full_tracking_df_list.append(current_halo['tracking.df'])
            t_e_halo = time.time()
            utilities.print_time_taken(t_s_halo, t_e_halo, "*" ,True, out_f)
    # Make a subhalo catalog using the tracked data for each of the subhalos.
    t_s_catalog = time.time()
    print(f"  * Making a subhalo catalog using the tracking data... ", end="", flush=True, file=out_f)
    subhalo_catalog_df = make_subhalo_catalog(BH_parameters, sim_num, full_tracking_df_list, out_f)
    ###### test
    #pd.set_option('display.width', 200)
    #print(subhalo_catalog_df)
    # Save the subhalo catalog as a .csv file.
    subhalo_catalog_outfname = f"{BH_parameters['subhalo_catalog_dir']}/subhalo_catalog.csv"
    subhalo_catalog_df.to_csv(subhalo_catalog_outfname, index=False)
    t_e_catalog = time.time()
    utilities.print_time_taken(t_s_catalog, t_e_catalog, "*" ,True, out_f)
    # Convert the list of dataframes to one dataframe.
    df_tracking_data = pd.concat(full_tracking_df_list)
    # Save the tracking data dataframe as a .hdf5 file.
    tracking_data_outfname = f"{BH_parameters['subhalo_tracking_dir']}/subhalo_tracking_data.hdf5"
    df_tracking_data.to_hdf(tracking_data_outfname, key='df', mode='w')
    print("", flush=True, file=out_f)
    print(f"  * Subhalo analysis completed!", flush=True, file=out_f)
    print(f"    * Subhalo catalog is saved at: {subhalo_catalog_outfname}", flush=True, file=out_f)
    print(f"    * Tracking data is saved at: {tracking_data_outfname}", flush=True, file=out_f)
#
#-----------------------------------------------------------------------------------
### Main function ###
# Steps 1--4 can be done separately.
#-----------------------------------------------------------------------------------
def main():
    t_s = time.time()
    #
    # Initialization process.
    # BH_initialization defines two global variables (COSMO and T0), in addition to returned variables.
    header_statement = "Bloodhound subhalo tracking progress report"
    BH_parameters, sim_nums, base_dir, out_f, snapnum_info_dict = BH_initialization(parameter_fname, header_statement)
    '''
    * Step 1: Merger tree data pre-processing
    '''
    # Check the user input in the parameter file to see whether the pre-processing step needs to be done or not.
    do_tree_processing = BH_parameters['do_tree_processing']
    if do_tree_processing == 1:
        t_s_step = time.time()
        print("####### do_tree_processing = 1: making various halo main-branch and subhalo catalog data... #######", flush=True, file=out_f)
        tree_pre_processing.main(BH_parameters, out_f)
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "#######", True, out_f)
        print("", flush=True, file=out_f)
    elif do_tree_processing == 0:
        print("####### do_tree_processing = 0: skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    '''
    * Step 2: Identifying infalling subhalos
    '''
    #
    # Check the user input in the parameter file to see whether the subhalo selection step needs to be done or not.
    do_subhalo_selection = BH_parameters['do_subhalo_selection']
    if do_subhalo_selection == 1:
        # If it hasn't been done yet, do it now.
        t_s_step = time.time()
        print("####### do_subhalo_selection = 1 #######", flush=True, file=out_f)
        print("* Identifying infalling subhalos from the merger tree data, for tracking...", flush=True, file=out_f)
        print("* Its progress will be recorded in a separate text file.", flush=True, file=out_f)
        infall_subhalo_criteria.main()
        t_e_step = time.time()
        utilities.print_time_taken(t_s_step, t_e_step, "#######", True, out_f)
        print("", flush=True, file=out_f)
    elif do_subhalo_selection == 0:
        print("####### do_subhalo_selection = 0: skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    '''
    * Step 3: subhalo particle tracking
    '''
    #
    # Check the user input in the parameter file to see whether the subhalo particle tracking step needs to be done or not.
    do_halo_particle_tracking = BH_parameters['do_halo_particle_tracking']
    if do_halo_particle_tracking == 1:
        t_s_tracking = time.time()
        print("####### do_halo_particle_tracking == 1: starting subhalo particle tracking #######", flush=True, file=out_f)
        for sim_num in sim_nums:
            t_s_sim = time.time()
            print(f"##### Simulation {sim_num} #####", flush=True, file=out_f)
            print("", flush=True, file=out_f)
            #
            # Initialize halo tracking.
            t_s_step = time.time()
            print("### Initializing halo tracking ###", flush=True, file=out_f)
            if BH_parameters['simulation_name'] == 'pELVIS':
                infall_information_dict = initialize_halo_tracking(BH_parameters, sim_num, out_f)
            elif BH_parameters['simulation_name'] == 'FIRE':
                infall_information_dict = initialize_halo_tracking_FIRE(BH_parameters, sim_num, out_f)
            t_e_step = time.time()
            print("### Initializing halo tracking finished! ###", flush=True, file=out_f)
            utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
            print("", flush=True, file=out_f)
            #
            # Perform subhalo tracking.
            t_s_step = time.time()
            print("### Tracking subhalos: one snapshot at a time ###", flush=True, file=out_f)
            subhalo_tracking_wrapper_function(BH_parameters, sim_num, infall_information_dict, out_f)
            t_e_step = time.time()
            print("### Subhalo tracking for current simulation finished! ###", flush=True, file=out_f)
            utilities.print_time_taken(t_s_step, t_e_step, "###", True, out_f)
            print("", flush=True, file=out_f)
            #
            #
            t_e_sim = time.time()
            print(f"##### Simulation {sim_num} finished! #####", flush=True, file=out_f)
            utilities.print_time_taken(t_s_sim, t_e_sim, "#####", True, out_f)
            print("", flush=True, file=out_f)
        t_e_tracking = time.time()
        print("####### Subhalo tracking finished! #######", flush=True, file=out_f)
        utilities.print_time_taken(t_s_tracking, t_e_tracking, "#######", True, out_f)
        print("", flush=True, file=out_f)
        #
    elif do_halo_particle_tracking == 0:
        print("####### do_halo_particle_tracking == 0: skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    '''
    * Step 4: subhalo analysis
    '''
    #
    # Check the user input in the parameter file to see whether the subhalo analysis step needs to be done or not.
    do_subhalo_analysis = BH_parameters['do_subhalo_analysis']
    if do_subhalo_analysis == 1:
        t_s_analysis = time.time()
        print("####### do_subhalo_analysis == 1: computing subhalo properties using tracked particles #######", flush=True, file=out_f)
        for sim_num in sim_nums:
            subhalo_analysis_wrapper_function(BH_parameters, sim_num, snapnum_info_dict, out_f)
        t_e_analysis = time.time()
        print("####### Subhalo analysis finished! #######", flush=True, file=out_f)
        utilities.print_time_taken(t_s_analysis, t_e_analysis, "#######", True, out_f)
        #
    elif do_subhalo_analysis == 0:
        print("####### do_subhalo_analysis == 0: skip this step! #######", flush=True, file=out_f)
        print("", flush=True, file=out_f)
    #
    #
    # Print the total execution time and close the output statement file.
    print("", flush=True, file=out_f)
    t_e = time.time()
    print(f"########## Total execution time: {t_e - t_s:.03f} s ##########", flush=True, file=out_f)
    #
    out_f.close()
#
if __name__ == "__main__":
    main()
