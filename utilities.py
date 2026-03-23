####################################################################################
# Import libraries.
####################################################################################
import h5py
import numpy as np
import os
import pandas as pd
import pathlib
import re
import time
#
####################################################################################
# Classes
####################################################################################
#
'''
Utilities to read in simulation snapshot data.
GIZMO:
  - Gas: particle type=0
  - Halo dark matter/collisionless particles: type=1
  - Disk collisionless particles: type=2
  - Bulge collisionless particles: type=3
    -- 2,3 = 'dummy' collisionless particles (e.g. low-res dark-matter particles in cosmological runs; pre-existing "disk" and "bulge" star particles in the non-cosmological runs, dust grains or cosmic ray particles in simulations with explicit grain dynamics or MHD-PIC simulations)
  - Stars stars spawned from gas: type=4
  - Bndry black holes (if active), or collisionless: type=5
  *** From the GIZMO documentation:
      - The names 'Gas', 'Halo', 'Disk' ... are just arbitrary conventions.
  *** For Phat ELVIS:
      - PartType1 = dark matter high-res?
'''
class SnapshotData_FIRE(dict):
    '''
    * A dictionary-class for handling snapshot data
    - For now, only compatible with the Phat ELVIS simulations
    - The original Phat ELVIS simulations have rather inconsistent snapshot file names,
      so this class handles them in very specific ways.
    - For one snapshot
    '''
    def __init__(self, sim_num, snap_num, base_dir, blocks):
        self['simulation.number'] = sim_num
        self['snapshot.number'] = snap_num
        self['base.directory'] = base_dir
        self['number.of.blocks'] = blocks
        #
        # Make snapshot file names.
        self['file.path'] = []
        if blocks == 1:
            snap_fname = f"{base_dir}/output/snapshot_{snap_num:03d}.hdf5"
            self['file.path'].append(snap_fname)
        elif blocks > 1:
            for i in range(blocks):
                block_fname = f"{base_dir}/output/snapdir_{snap_num:03d}/snapshot_{snap_num:03d}.{i}.hdf5"
                self['file.path'].append(block_fname)
                
    #
    def read_in_pID_argsort_data(self):
        self['file.path.sort_idx'] = f"{self['base.directory']}/{self['run.type']}/halo_{self['simulation.number']}/sorted_pID/part_ID_argsort_idx_{self['snapshot.number']:03d}.npz"
        #
        # Open the particle ID sort index file.
        with np.load(self['file.path.sort_idx'], mmap_mode='r', allow_pickle=False) as sort_idx_file:
            self['pID.sort_idx'] = sort_idx_file['particle_ID']
    #
    def read_in_snapshot_data(self, use_argsort):
        '''
        * This method opens and reads in the snapshot data for one snapshot.
        - If the snapshot data is stored in multiple files, it will concatenate the data.
        '''
        for i in range(self['number.of.blocks']):
            snap_fname = self['file.path'][i]
            #
            # Open the snapshot block.
            with h5py.File(snap_fname, 'r') as snap_dat:
                #
                # Read in Particle IDs, Coordinates, and Velocities.
                if use_argsort:
                    # If particle ID argsort data exists, don't need to read in the particle ID data from the simulation output.
                    coordinates = snap_dat['PartType1/Coordinates'][:]
                    velocities = snap_dat['PartType1/Velocities'][:]
                    #particle_IDs = snap_dat['PartType1/ParticleIDs'][:]
                    #
                    # Append the data to the final array.
                    if i == 0:
                        #pID_arr = particle_IDs
                        coord_arr = coordinates
                        vel_arr = velocities
                    else:
                        #pID_arr = np.concatenate((pID_arr, particle_IDs))
                        coord_arr = np.concatenate((coord_arr, coordinates), axis=0)
                        vel_arr = np.concatenate((vel_arr, velocities), axis=0)
                    
                else:
                    particle_IDs = snap_dat['PartType1/ParticleIDs'][:]
                    coordinates = snap_dat['PartType1/Coordinates'][:]
                    velocities = snap_dat['PartType1/Velocities'][:]
                    if 'PartType2' in list(snap_dat.keys()):
                        particle_IDs = np.concatenate((particle_IDs, snap_dat['PartType2/ParticleIDs'][:]))
                        coordinates = np.concatenate((coordinates, snap_dat['PartType2/Coordinates'][:]))
                        velocities = np.concatenate((velocities, snap_dat['PartType2/Velocities'][:]))
                    #
                    # Append the data to the final array.
                    if i == 0:
                        pID_arr = particle_IDs
                        coord_arr = coordinates
                        vel_arr = velocities
                    else:
                        pID_arr = np.concatenate((pID_arr, particle_IDs))
                        coord_arr = np.concatenate((coord_arr, coordinates), axis=0)
                        vel_arr = np.concatenate((vel_arr, velocities), axis=0)
        #
        # Store the merged data.
        self["Coordinates"] = coord_arr
        self["Velocities"] = vel_arr
        #self["ID.particle"] = pID_arr
        if not use_argsort:
            # Sort the particle ID array and store the indices that would sort it.
            self['pID.sort_idx'] = np.argsort(pID_arr)
            self["ID.particle"] = pID_arr
        # Store the smallest particle ID value
        self['ID.particle.min'] = np.min(pID_arr)
    #
    def print_header(self, out_f):
        fname = self['file.path'][0]
        with h5py.File(fname, 'r') as snap_dat:
            print(f"  * Groups: {list(snap_dat.keys())}", flush=True, file=out_f)
            print(f"  * Members of 'PartType0': {list(snap_dat['PartType0'].keys())}", flush=True, file=out_f)
            print(f"  * Members of 'PartType1': {list(snap_dat['PartType1'].keys())}", flush=True, file=out_f)
            print(f"  * Metadata dictionary: {dict(snap_dat['Header'].attrs.items())}", flush=True, file=out_f)
class SnapshotData(dict):
    '''
    * A dictionary-class for handling snapshot data
    - For now, only compatible with the Phat ELVIS simulations
    - The original Phat ELVIS simulations have rather inconsistent snapshot file names,
      so this class handles them in very specific ways.
    - For one snapshot
    '''
    def __init__(self, sim_num, snap_num, base_dir, run_type, blocks):
        self['simulation.number'] = sim_num
        self['snapshot.number'] = snap_num
        self['base.directory'] = base_dir
        self['run.type'] = run_type
        self['number.of.blocks'] = blocks
        #
        # Make snapshot file names.
        self['file.path'] = []
        if run_type == 'dmo':
            if blocks == 1:
                snap_fname = f"{base_dir}/{run_type}/halo_{sim_num}/output/zoom_Z13_z125_{snap_num:03d}.hdf5"
                self['file.path'].append(snap_fname)
            elif blocks > 1:
                for i in range(blocks):
                    block_fname = f"{base_dir}/{run_type}/halo_{sim_num}/z13/snapdir_{snap_num:03d}/zoom_Z13_z125_{snap_num:03d}.{i}.hdf5"
                    self['file.path'].append(block_fname)
        elif run_type =='disk':
            if blocks == 1:
                if sim_num == 848:
                    # Simulation 848 is the only Disk simulation that doesn't have the sim_num in the snapshot name.
                    snap_fname = f"{base_dir}/{run_type}/halo_{sim_num}/output/zoom_Z13_z125_{run_type}_{snap_num:03d}.hdf5"
                else:
                    snap_fname = f"{base_dir}/{run_type}/halo_{sim_num}/output/zoom_Z13_z125_{sim_num}_{run_type}_{snap_num:03d}.hdf5"
                self['file.path'].append(snap_fname)
            elif blocks > 1:
                for i in range(blocks):
                    block_fname = f"{base_dir}/{run_type}/halo_{sim_num}/z13/snapdir_{snap_num:03d}/zoom_Z13_z125_{run_type}_{snap_num:03d}.{i}.hdf5"
                    self['file.path'].append(block_fname)
    #
    def read_in_pID_argsort_data(self):
        self['file.path.sort_idx'] = f"{self['base.directory']}/{self['run.type']}/halo_{self['simulation.number']}/sorted_pID/part_ID_argsort_idx_{self['snapshot.number']:03d}.npz"
        #
        # Open the particle ID sort index file.
        with np.load(self['file.path.sort_idx'], mmap_mode='r', allow_pickle=False) as sort_idx_file:
            self['pID.sort_idx'] = sort_idx_file['particle_ID']
    #
    def read_in_snapshot_data(self, use_argsort):
        '''
        * This method opens and reads in the snapshot data for one snapshot.
        - If the snapshot data is stored in multiple files, it will concatenate the data.
        '''
        for i in range(self['number.of.blocks']):
            snap_fname = self['file.path'][i]
            #
            # Open the snapshot block.
            with h5py.File(snap_fname, 'r') as snap_dat:
                #
                # Read in Particle IDs, Coordinates, and Velocities.
                if use_argsort:
                    # If particle ID argsort data exists, don't need to read in the particle ID data from the simulation output.
                    coordinates = snap_dat['PartType1/Coordinates'][:]
                    velocities = snap_dat['PartType1/Velocities'][:]
                    #particle_IDs = snap_dat['PartType1/ParticleIDs'][:]
                    #
                    # Append the data to the final array.
                    if i == 0:
                        #pID_arr = particle_IDs
                        coord_arr = coordinates
                        vel_arr = velocities
                    else:
                        #pID_arr = np.concatenate((pID_arr, particle_IDs))
                        coord_arr = np.concatenate((coord_arr, coordinates), axis=0)
                        vel_arr = np.concatenate((vel_arr, velocities), axis=0)
                    
                else:
                    particle_IDs = snap_dat['PartType1/ParticleIDs'][:]
                    coordinates = snap_dat['PartType1/Coordinates'][:]
                    velocities = snap_dat['PartType1/Velocities'][:]
                    #
                    # Append the data to the final array.
                    if i == 0:
                        pID_arr = particle_IDs
                        coord_arr = coordinates
                        vel_arr = velocities
                    else:
                        pID_arr = np.concatenate((pID_arr, particle_IDs))
                        coord_arr = np.concatenate((coord_arr, coordinates), axis=0)
                        vel_arr = np.concatenate((vel_arr, velocities), axis=0)
        #
        # Store the merged data.
        self["Coordinates"] = coord_arr
        self["Velocities"] = vel_arr
        #self["ID.particle"] = pID_arr
        if not use_argsort:
            self["ID.particle"] = pID_arr
    #
    def print_header(self, out_f):
        fname = self['file.path'][0]
        with h5py.File(fname, 'r') as snap_dat:
            print(f"  * Groups: {list(snap_dat.keys())}", flush=True, file=out_f)
            print(f"  * Members of 'PartType0': {list(snap_dat['PartType0'].keys())}", flush=True, file=out_f)
            print(f"  * Members of 'PartType1': {list(snap_dat['PartType1'].keys())}", flush=True, file=out_f)
            print(f"  * Metadata dictionary: {dict(snap_dat['Header'].attrs.items())}", flush=True, file=out_f)
#
####################################################################################
# Functions
####################################################################################
def open_halo_particles_file(file_name):
    '''
    * Function to open halo particle files.
    - Returns snapshot numbers, particle coordinates, velocities, and particle IDs.
    '''
    # Open file
    with h5py.File(file_name, 'r') as f:
        keys = list(f.keys())
        # Lists to append data to.
        snap_list = []
        coords_list = []
        vels_list = []
        pIDs_list = []
        masses_list = []
        # Read in each snapshot and append data to lists.
        for i in range(len(keys)):
            snapnum = int(keys[i].split('_')[1])
            snap_dat = f[keys[i]]
            if i == 0:
                pIDs = snap_dat['ParticleIDs'][:]
            # Get coordinates, velocities, and masses.
            coords = snap_dat['Coordinates'][:]
            vels = snap_dat['Velocities'][:]
            #masses = snap_dat['Masses'][:]
            # Append to lists.
            snap_list.append(snapnum)
            coords_list.append(coords)
            vels_list.append(vels)
            #masses_list.append(masses)
        # Sort by snapshot number
        argsort_idx = np.argsort(snap_list)
        snap_arr_sorted = np.array(snap_list)[argsort_idx]
        coords_arr = np.array(coords_list)[argsort_idx]
        vels_arr = np.array(vels_list)[argsort_idx]
        # Return results.
        return(snap_arr_sorted, coords_arr, vels_arr, pIDs)
#
def get_halo_particle_file_names_in_dir(dpath):
    '''
    * Function to obtain halo particle file names.
    - Returns 3 lists containing halo IDs, infall snapshot numbers, and full file names, sorted by halo IDs.
    '''
    #
    # Get all file names in the directory.
    _, _, fnames = next(os.walk(dpath))
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
        full_name = pathlib.PurePath(dpath, fname)
        # hID_(hID)_infall_snap_(infall snap)_particles.hdf5
        #
        # Get halo ID and infall snapshot number from file names.
        hID = int(fname.split('_')[1])
        infall_snap = int(fname.split('_')[4])
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
#
def output_halo_particles_hdf5(BH_parameters, sim_num, halo_ID, infall_snap, snaps_to_save,
                               infall_halo_pIDs, tracked_particle_data, run_type, out_f):
    dir_name = BH_parameters["tracked_halo_particle_dir"]
    '''
    * This function saves tracked halo particle data for a single halo in a .hdf5 file.
    - dir_name: base directory path for the output .hdf5 file
    - sim_num: simulation number for the current halo
    - halo_ID: halo ID of the halo
    - infall_snap: infall snapshot number for the current halo, used in the file name.
    - snaps_to_save: a list of snapshot numbers for which the particle data need to be saved.
                     For default Bloodhound, it is a list containing a single snapshot number.
    - infall_halo_pIDs: particle IDs of the current halo at its infall snapshot
    - tracked_particle_data: a list of list [coordinates_array, velocities_array] for the current halo at the snapshot(s) in snaps_to_save.
    - run_type: simulation type - For Phat ELVIS, 'dmo' or 'disk'
    '''
    # Make the output file name.
    if BH_parameters['simulation_name'] == 'pELVIS':
        fname = f"{dir_name}/{run_type}/{sim_num}/hID_{halo_ID}_infall_snap_{infall_snap}_particles.hdf5"
    elif BH_parameters['simulation_name'] == 'FIRE':
        fname = f"{dir_name}/hID_{halo_ID}_infall_snap_{infall_snap}_particles.hdf5"
    #
    # Make dataset names.
    dset_name_pID = "ParticleIDs"
    dset_name_coords = "Coordinates"
    dset_name_vels = "Velocities"
    #
    # Open the output file.
    with h5py.File(fname, 'a') as out_hf:
        # One group for each snapshot
        for i in range(len(tracked_particle_data)):
            current_snap = snaps_to_save[i]
            particle_data = tracked_particle_data[i]
            coords = particle_data[0]
            vels = particle_data[1]
            #
            # Open a group for the current snapshot, creating it if it doesn't exist.
            grp = out_hf.require_group(f'snapshot_{current_snap}')
            #
            # Check and create datasets - ParticleIDs, Coordinates, and Velocities.
            # If the dataset already exists, skip saving.
            # I could loop over ParticleIDs, Coordinates, and Velocities but I chose not to.
            # Particle IDs
            if dset_name_pID not in grp:
                grp.create_dataset(dset_name_pID, data=infall_halo_pIDs)
            #else:
                #print(f"    * Halo {halo_ID}: dataset /snapshot_{current_snap}/{dset_name_pID} already exists and saving is skipped.", flush=True, file=out_f)
            # Coordinates
            if dset_name_coords not in grp:
                grp.create_dataset(dset_name_coords, data=coords)
            #else:
                #print(f"    * Halo {halo_ID}: dataset /snapshot_{current_snap}/{dset_name_coords} already exists and saving is skipped.", flush=True, file=out_f)
            # Velocities
            if dset_name_vels not in grp:
                grp.create_dataset(dset_name_vels, data=vels)
            #else:
                #print(f"    * Halo {halo_ID}: dataset /snapshot_{current_snap}/{dset_name_vels} already exists and saving is skipped.", flush=True, file=out_f)
#
def make_rockstar_fnames(fdir, num_files, snapshot_num, file_type):
    '''
    * This function creates the full file path/name for the given Rockstar file type (.ascii or .bin).
    - num_files: number of files the data is divided into per snapshot.
    '''
    fname_list = []
    fname_base = f'{fdir}/halos_{snapshot_num:03d}'
    #
    # Create file names.
    for i in range(num_files):
        file_name = f"{fname_base}.{i}.{file_type}"
        fname_list.append(file_name)
    return(fname_list)
#
def write_header_for_result_text_file(header_statement, out_f):
    current_time = get_current_time(for_fname=False)
    curremt_time_str = f"Date: {current_time}"
    #
    # Set num_char equal to the length of the longer between header_statement and curremt_time_str.
    if len(header_statement) > len(curremt_time_str):
        num_char = len(header_statement)
    elif len(header_statement) < len(curremt_time_str):
        num_char = len(curremt_time_str)
    else:
        num_char = len(header_statement)
    border = "#"*(num_char + 16)
    header_statement_line = f"####### {header_statement: <{num_char}} #######"
    time_line = f"####### {curremt_time_str: <{num_char}} #######"
    print(border, flush=True, file=out_f)
    print(header_statement_line, flush=True, file=out_f)
    print(time_line, flush=True, file=out_f)
    print(border, flush=True, file=out_f)
    print("", flush=True, file=out_f)
'''
* A simple function to print input parameter values in the parameter file.
'''
def print_params(BH_parameters, out_f):
    #params_to_print = ['h', 'omega_l', 'omega_m', 'temp_cmb', 'G', 'part_mass', 'subhalo_selection_done', 'halo_particle_tracking_done', 'first_infall_z_high', 'first_infall_z_low', 'min_vinfall', 'max_vinfall', 'two_rockstars', 'last_snapnum', 'tracking_order', 'pID_argsort_made']
    params_to_print = ['h', 'omega_l', 'omega_m', 'temp_cmb', 'G', 'part_mass', 'do_tree_processing', 'do_subhalo_selection', 'do_halo_particle_tracking', 'do_subhalo_analysis', 'first_infall_z_high', 'first_infall_z_low', 'min_vinfall', 'max_vinfall', 'two_rockstars', 'last_snapnum', 'tracking_order', 'pID_argsort_made']
    dirs_to_print = ['base_dir', 'snapnum_info_fname', 'out_statement_dir', 'bloodhound_out_statement_fname_base', 'infall_criteria_out_statement_fname_base', 'infall_subtree_out_dir', 'infall_subtree_out_fname_base', 'tracked_halo_particle_dir', 'subhalo_tracking_dir', 'subhalo_catalog_dir', 'tree_hdf5_fname', 'tree_processed_data_out_dir']
    print("####### Input parameters from the parameter file #######", flush=True, file=out_f)
    #
    # Print parameter key: value pairs.
    print("* ", end="", flush=True, file=out_f)
    for i in range(len(params_to_print)):
        key = params_to_print[i]
        if i != len(params_to_print)-1:
            print(f"{key}: {BH_parameters[key]}, ", end="", flush=True, file=out_f)
        else:
            print(f"{key}: {BH_parameters[key]}, ", flush=True, file=out_f)
    #
    # Print directory/file paths.
    for path in dirs_to_print:
        print(f"* {path}: {BH_parameters[path]}", flush=True, file=out_f)
    #
    # Pring simulation type.
    #print(f"* Infalling subhalos are identified from <<{BH_parameters['run_type']}>> simulations.", flush=True, file=out_f)
    #
    # Print simulation numbers.
    print(f"* Simulations to use: {BH_parameters['sim_nums']}", flush=True, file=out_f)
    #
    # Print whether snapshot particle ID argsort files exist.
    if BH_parameters['pID_argsort_made'] == True:
        print("* Snapshot particle ID argsort files have been made, so particle tracking will be done using them.", flush=True, file=out_f)
    else:
        print("* Snapshot particle ID argsort files have not been made, so particle tracking will be done using only the snapshot output data", flush=True, file=out_f)
    print("", flush=True, file=out_f)
#
def get_current_time(for_fname):
    # Get the current time in seconds since the epoch.
    current_time_seconds = time.time()
    # Convert the current time to a tuple representing local time.
    local_time_tuple = time.localtime(current_time_seconds)
    # Format the local time tuple into a human-readable string.
    if for_fname == True:
        formatted_time = time.strftime("%Y_%m_%d_%H_%M_%S", local_time_tuple)
    else:
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", local_time_tuple)
    return(formatted_time)
#
'''
* A function to initialize a generic output statement file.
'''
def open_output_statement_file(file_dir, base_name):
    # get_current_time(for_fname)
    # Get today's date and time to use for the file name.
    date_time = get_current_time(for_fname=True)
    #
    # Make the file name.
    fname = f"{file_dir}/{base_name}_{date_time}.txt"
    #
    # Open the output statement file.
    out_f = open(fname, 'w')
    return(out_f)
#
def print_time_taken(t_s, t_e, text_decor, to_file, out_f):
    statement = f"{text_decor} Execution time: {t_e - t_s:.03f} s {text_decor}"
    if to_file == True:
        print(statement, flush=True, file=out_f)
    else:
        print(statement)
#
'''
* Check if the given string is an integer.
'''
def is_integer(string):
    try:
        int(string)
        return True
    except ValueError:
        return False
#
'''
* Check if the given string is a float.
'''
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False
#
'''
* Convert a string in scientific notation to a float.
'''
def convert_scientific_to_float(scientific_notation):
    try:
        return float(scientific_notation)
    except ValueError:
        parts = scientific_notation.split('e')
        coefficient = float(parts[0])
        exponent = int(parts[1])
        return coefficient * (10 ** exponent)
#
'''
* Convert a string list to a list of numbers.
  - string_list is the list in the string format: "[1,2,3]"
'''
def convert_string_list_to_list(string_list):
    # Initialize the result list.
    result_list = []
    # Get the list elements as strings.
    numbers_as_strings = string_list.strip('[]').split(', ')
    for num in numbers_as_strings:
        # If an integer, convert to an integer.
        if is_integer(num):
            num = int(num)
        # If a float (or scientific notation), convert to a float.
        elif is_float(num):
            # Convert scientific notations to floats.
            if 'e' in num.lower():
                num = convert_scientific_to_float(num)
            # Convert to a float.
            else:
                num = float(num)
        else:
            # Assume everything else is a string.
            num = num
        #
        # Append the element to the result list.
        result_list.append(num)
    return(result_list)
#
'''
* Convert a string 'True', 'true', 'False', or 'false' to their respective boolean values.
'''
def convert_to_boolean(value):
    if value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    else:
        return(value)  # Return the original value if it's not 'true' or 'false'
'''
* Read parameters from bloodhound_parameters.txt file.
'''
def read_parameters(fname):
    # Initialize the result dictionary.
    parameters = {}
    #
    # Open the parameter file and read-in the data.
    with open(fname, 'r') as f:
        for line in f:
            # Ignore comments and empty lines
            if not line.strip() or line.strip().startswith('#'):
                continue
            # Split line into parameter name and value
            parts = line.split('=')
            if len(parts) == 2:
                parameter_name = parts[0].strip()
                parameter_value = parts[1].split('#')[0].strip()
                if is_integer(parameter_value):
                    # Convert integers.
                    parameter_value = int(parameter_value)
                elif is_float(parameter_value):
                    # Convert scientific notations to floats.
                    if 'e' in parameter_value.lower():
                        parameter_value = convert_scientific_to_float(parameter_value)
                    # Convert floats
                    else:
                        parameter_value = float(parameter_value)
                else:
                    # Remove redundant string markers.
                    if parameter_value.startswith('"') or parameter_value.startswith("'"):
                        parameter_value = parameter_value[1:-1]
                    # Handle f-strings.
                    elif parameter_value.startswith('f"') or parameter_value.startswith("f'"):
                        parameter_value = eval(parameter_value, {}, parameters)
                    # Convert lists.
                    elif parameter_value.startswith('[') and parameter_value.endswith(']'):
                        parameter_value = convert_string_list_to_list(parameter_value)
                    #
                    else:
                        # Convert boolean strings (i.e. 'true' or 'False' etc) to boolean values.
                        # If not a boolean string, convert_to_boolean(value) returns the original value.
                        parameter_value = convert_to_boolean(parameter_value)
                # Assign the parameter value to the parameter key.
                parameters[parameter_name] = parameter_value
    return(parameters)
#
'''
* Read-in the snapshot number, redshift, and scale factor data.
'''
def open_snap_header_file(fname, sim_name):
    # Initialize result lists.
    full_snap_list = []
    full_z_list = []
    full_a_list = []
    #full_t_list = []
    #full_t_width_list = []
    # Open the file and read-in the data.
    with open(fname) as f:
        for line in f:
            if line[0] == "#":
                # Skip comment lines.
                continue
            line_list = line.split()
            if sim_name == 'pELVIS':
                # For Phat ELVIS, the snapshot time file contains:
                # snapshot number, redshift, scale factor.
                full_snap_list.append(int(line_list[0]))
                full_z_list.append(float(line_list[1]))
                full_a_list.append(float(line_list[2]))
            elif sim_name == 'FIRE' or sim_name == 'fire':
                # For FIRE, the snapshot time file contains:
                # snapshot number (i), scale factor, redshift, time [Gyr], time width (Myr)
                full_snap_list.append(int(line_list[0]))
                full_a_list.append(float(line_list[1]))
                full_z_list.append(float(line_list[2]))
                #full_t_list.append(float(line_list[3]))
                #full_t_width_list.append(float(line_list[4]))
    #
    # Convert the lists to arrays.
    full_snap_arr = np.array(full_snap_list)
    full_z_arr = np.array(full_z_list)
    full_a_arr = np.array(full_a_list)
    #full_t_arr = np.array(full_t_list)
    #full_t_width_arr = np.array(full_t_width_list)
    #
    # Make a result dictionary.
    if sim_name == 'pELVIS':
        header_dict = {
            "snapshot_numbers": full_snap_arr,
            "redshifts": full_z_arr,
            "scale_factors": full_a_arr
        }
    elif sim_name == 'FIRE' or sim_name == 'fire':
        header_dict = {
            "snapshot_numbers": full_snap_arr,
            "redshifts": full_z_arr,
            "scale_factors": full_a_arr,
            #"time": full_t_arr,
        }

    return(header_dict)
'''
* A function to make and return file names.
- host main branch file
- merger tree main branch file
- subtree main branch file
- disk infall subtree file
- identified DMO tree/subtree file
- subid-tid pair file
- tracked subhalo property file


#
# File path for the Disk subtree main branch file.
subtree_fpath = f'{base_dir}/{simnum}_{simtype}_host_subtree_main_branch.hdf5'
#
# File paths for DMO surviving halo main branch trees.
if simnum==493:
    surv_tree_fpath = f'{base_dir}/main_branches_dmo_{simnum}_tyler.csv'
else:
    surv_tree_fpath = f'{base_dir}/main_branches_dmo_{simnum}_new.csv'
#
# File paths for DMO subtree main branch file.
dest_tree_fpath = f'{base_dir}/{simnum}_dmo_host_subtrees_main_branch_new.hdf5'
#
# File path for the Disk subtree file with subhalos found by using the infall criteria.
infall_sub_fpath = f'{base_dir}/{simnum}_{simtype}_subtrees_infall_criteria.hdf5'
#
# File path for the identified DMO subtree/trees.
iden_dmo_tree_fpath = f'{base_dir}/{simnum}_{simtype_dmo}_subtrees_matched_stampede.hdf5'
#
# File paths for tracked subhalo particles.
disk_part_dir = f'{base_dir}/massive_subs/{simtype}_1'
dmo_part_dir = f'{base_dir}/massive_subs/{simtype_dmo}_1'
#
# File paths for the subid-hid pair path
subid_hid_pair_path = f'/scratch/05097/hk9457/hyunsu/pELVIS/z13/disk_from_ranch/halo_{simnum}/particle_tracking/{simnum}_disk_infall_criteria_tid_hid_pairs.csv'
'''
def various_halo_file_names(base_dir, sim_num):
    # Initialize a dictionary to return with all file names.
    all_fname_dict = {}
    '''
    * Set file names into the dictionary.
    '''
    base_dir = f"{base_dir}/disk/halo_{sim_num}/subhalo_analysis"
    # Host main branch file.
    all_fname_dict["host_main_branch_dmo"] = f'{base_dir}/{sim_num}_dmo_host_main_branch_new.csv'
    all_fname_dict["host_main_branch_disk"] = f'{base_dir}/{sim_num}_disk_host_main_branch_new.csv'
    #
    # (Surviving) merger tree main branch file.
    # For now, DMO only.
    #
    if sim_num==493:
        all_fname_dict["main_branch_dmo"] = f"{base_dir}/main_branches_dmo_{sim_num}_tyler.csv"
    else:
        all_fname_dict["main_branch_dmo"] = f"{base_dir}/main_branches_dmo_{sim_num}_new.csv"
    ############################################
    # Completed only to here!
    ############################################
    #
    # (Destroyed) subtree main branch file.
    all_fname_dict["subtree_dmo"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/subtrees_from_host/{sim_num}_dmo_host_subtrees_main_branch_new.hdf5"
    all_fname_dict["subtree_disk"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/subtrees_from_host/{sim_num}_disk_host_subtrees_main_branch_new.hdf5"
    #
    # Disk subtree infall criteria file.
    all_fname_dict["infall_subtree_disk"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/subtrees_from_host/{sim_num}_disk_subtrees_infall_criteria.hdf5"
    #
    # Identified DMO tree/subtree file: out of all subhalos in "infall_subtree_disk".
    all_fname_dict["iden_dmo_tree"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/subtrees_from_host/{sim_num}_dmo_subtrees_matched_merged.hdf5"
    #
    # Disk subID-tID pair file: for subhalos in "infall_subtree_disk"
    all_fname_dict["subID_hID_pair"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/tID_hID_pairs/{sim_num}_disk_infall_criteria_tid_hid_pairs.csv"
    #
    # tracked subhalo property file
    all_fname_dict["tracked_subhalo_property_dmo"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/tracked_subhalo_properties/from_disk_subtrees/{sim_num}_tree_from_tracked_particles_dmo_merged.hdf5"
    all_fname_dict["tracked_subhalo_property_disk"] = f"{base_dir}pELVIS_z13_halo_tracking/{sim_num}/tracked_subhalo_properties/from_disk_subtrees/{sim_num}_tree_from_tracked_particles_disk_merged.hdf5"
    #
    # Return the final dictionary.
    return(all_fname_dict)
#
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
#
'''
* Read-in the main branch data for the host halo.
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
    host_vx = np.flip(host_main_branch_file.vx.values)# I am pretty sure the velocities in rockstar/consistent-trees are in physical units (km/s), so no need to multiply it by sqrt(scale factor): snapshot data velocities are in comoving km/s.
    host_vy = np.flip(host_main_branch_file.vy.values)
    host_vz = np.flip(host_main_branch_file.vz.values)
    host_snapnums = np.flip(host_main_branch_file.snapshot.values)
    host_vmax = np.flip(host_main_branch_file['vel.circ.max'].values)
    host_rvir = np.flip(host_main_branch_file.radius.values) # physical kpc, it might be in COMOVING kpc!
    host_scales = np.flip(host_main_branch_file['scale.factor'].to_numpy())
    #host_rvir_pkpc = np.multiply(host_rvir, host_scale)
    #host_tid = np.flip(host_main_branch_file.id.values)
    # Get the time information for the simulation.
    #sim_snapshot_numbers = BH_parameters['time_info_dict']['snapshot_numbers']
    #sim_scale_factors = BH_parameters['time_info_dict']['scale_factors']
    #sim_redshifts = BH_parameters['time_info_dict']['redshifts']
    #sim_t_cosmic = BH_parameters['time_info_dict']['time']
    #sim_t_lookback = sim_t_cosmic[-1] - sim_t_cosmic
    # Match the snapshots to get the time information for the host halo tree.
    #first_idx = np.nonzero(np.isclose(sim_snapshot_numbers, host_snapnums[0]))[0][0]
    #host_scale = sim_scale_factors[first_idx:]
    #host_redshift = sim_redshifts[first_idx:]
    #host_t_cosmic = sim_t_cosmic[first_idx:]
    #host_t_lookback = sim_t_lookback[first_idx:]
    #
    # Create a dictionary for the host halo data.
    host_dict = {}
    host_dict['x'] = host_x
    host_dict['y'] = host_y
    host_dict['z'] = host_z
    host_dict['scale.factor'] = host_scales
    host_dict['vmax'] = host_vmax
    host_dict['rvir'] = host_rvir
    host_dict['vx'] = host_vx
    host_dict['vy'] = host_vy
    host_dict['vz'] = host_vz
    #host_dict['rvir_phys'] = host_rvir_pkpc
    #host_dict['tree_id'] = host_tid
    return(host_dict)
#
def read_in_host_main_branch_file(sim_file_name_dict, sim_num, sim_type, BH_parameters):
    cosmo = BH_parameters['cosmo']
    # All distance units will be converted to non-h units.
    h = BH_parameters['h']
    #
    # Make the dictionary key name for the subtree main branch file.
    fname_key = f"host_main_branch_{sim_type}"
    #
    # Get the file name.
    fname = sim_file_name_dict[fname_key]
    #
    # Read in the host main branch file for the current simulation.
    host_main_branch_file = pd.read_csv(fname)
    #
    # Get various properties of the host tree as arrays.
    # Flip the arrays so they are ordered from early time to late time.
    #np.flip(combined_arr, axis=0)
    #host_coordinates = np.flip(host_main_branch_file[['x','y','z']].values/h, axis=0) # Flipped vertically (axis=0)
    host_x = np.flip(host_main_branch_file.x.values/h)
    host_y = np.flip(host_main_branch_file.y.values/h)
    host_z = np.flip(host_main_branch_file.z.values/h)
    host_vx = np.flip(host_main_branch_file.vx.values)
    host_vy = np.flip(host_main_branch_file.vy.values)
    host_vz = np.flip(host_main_branch_file.vz.values)
    host_scale = np.flip(host_main_branch_file.scale.values)
    host_redshift = 1. / host_scale - 1.
    host_t_cosmic = cosmo.age(host_redshift).value
    host_t_lookback = cosmo.lookback_time(host_redshift).value
    host_vmax = np.flip(host_main_branch_file.vmax.values)
    if 'rvir' in host_main_branch_file.columns:
        host_rvir = np.flip(host_main_branch_file.rvir.values/h)
    elif 'Rvir' in host_main_branch_file.columns:
        host_rvir = np.flip(host_main_branch_file.Rvir.values/h)
    host_rvir_pkpc = np.multiply(host_rvir, host_scale)
    host_tid = np.flip(host_main_branch_file.id.values)
    #
    # Create a dictionary for the host halo data.
    host_dict = {}
    host_dict['x'] = host_x
    host_dict['y'] = host_y
    host_dict['z'] = host_z
    host_dict['vx'] = host_vx
    host_dict['vy'] = host_vy
    host_dict['vz'] = host_vz
    host_dict['scale'] = host_scale
    host_dict['vmax'] = host_vmax
    host_dict['rvir'] = host_rvir
    host_dict['rvir_phys'] = host_rvir_pkpc
    host_dict['tree_id'] = host_tid
    return(host_dict)

