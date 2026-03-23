#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 16:24:06 2020

@author: Hyunsu Kong <hyunsukong@utexas.edu>
"""

import os
import sys
import h5py
import time
import struct
import pathlib
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

#sys.path.insert(0, os.path.abspath('/Users/hk9457/Desktop/Hyunsu/Research/scripts'))
#sys.path.insert(0, os.path.abspath('/Users/hk9457/Desktop/Hyunsu/Research/scripts/wetzel'))
#from utilities.basic import io as ut_io

#----------------------------------------------------------------------------------------------------
# defaults
#----------------------------------------------------------------------------------------------------

# Planck 2016 comology for pELVIS simulations
omega_l=0.6879
omega_m=0.3121
h=0.6751
H0=0.1*h  # km / s / kpc
part_mass=1.9693723*1e4/h
G=4.30091*1e-6 # kpc / Msolar (km/s)^2


"""
- Make a .csv z=0 halo catalog using a Rockstar halo catalog output.

- For now, the "run script" (not necessarily this script) is assumed to be inside the 'scripts' directory inside the 'halo' folder.
(../simulation/halo/scripts/make_csv_catalog.py)
When I build a more extensive set of routines and scripts, the paths will eventually come to a more centralised location.

- The default output location is the same as the input location.

- 3 input parameters (halo_catalog_directory, snapshot_num, out_name):
    - halo_catalog_directory: directory where the catalog data is stored
        - should be "catalog"
    - snapshot_num: snapshot number to use
    - out_name: name of the .csv catalog to output
    
- Think about ways to handle .list files and .ascii files
"""



def make_csv_catalog(sim_num, catalog_directory_name, snapshot_num, out_dir, out_name):
    '''
    - catalog_directory_name: directory path to the Rockstar catalog results - input
    - snapshot_num: snapshot number to create the catalog for.
    - out_dir: directory path to output files.
    - out_name: name of the output catalog files.
    '''

    suffix='.csv'
    # get path to the halo directory
    #halo_directory = pathlib.Path.cwd().parent
    
    # combine halo_directory and halo_catalog_directory to get the full path to the directory
    #halo_catalog_directory = pathlib.PurePath(halo_directory, catalog_directory_name)
    halo_catalog_directory = catalog_directory_name    

    # path to the input halo catalog file
    in_catalog_name = f'out_{snapshot_num:03d}.list'
    full_halo_catalog_path = pathlib.PurePath(halo_catalog_directory, in_catalog_name)
    
    # path to the .csv catalog to output
    out_full_path = pathlib.PurePath(out_dir, out_name+suffix)
    
    print("* Rockstar halo catalog output file is:", full_halo_catalog_path)
    print()
    print("* Output catalog name/path is:", out_full_path)
    print()
    
    # read in the halo catalog
    catalog_file_df = pd.read_csv(full_halo_catalog_path, sep=" ", low_memory=False)
    # I just like things to be in lower-case
    catalog_file_df = catalog_file_df.rename(columns={'#ID':'orig_id', 'Mvir':'mvir', 'Rvir':'rvir','Vmax':'vmax','Rs':'rs','X':'x',
                         'Y':'y','Z':'z','VX':'vx','VY':'vy','VZ':'vz'})
    
    # Read in the scale factor: it probably will never be used
    snap_scale_factor=float(catalog_file_df.mvir[0])
    print(f'* Scale factor: {snap_scale_factor}')
    
    # The first 15 lines of data are simulation parameters/details and are read in as column entries
    # scale factor happens to fall under mvir
    # The 15 lines will be removed
    reduced_data=catalog_file_df[15:]
    # Make a smaller dataframe with things I need
    reduce_columns=reduced_data[['orig_id','mvir','vmax','rvir','rs','rs_klypin','x','y','z', 'vx', 'vy', 'vz']]
    reduce_columns=reduce_columns.astype(float)
    reduce_columns.orig_id=reduce_columns.orig_id.astype(int)
    
    # Get halos with v_max >= 4.5 km/s
    after_vmax=reduce_columns.query('vmax>=4.5')
    after_vmax['host_id']=sim_num
    after_vmax=after_vmax[['host_id', 'orig_id','mvir','vmax','rvir','rs','rs_klypin','x','y','z', 'vx', 'vy', 'vz']]
    
    # Convert all units to non-/h units by dividing length and mass units by h
    after_vmax[['mvir','rvir','rs','rs_klypin','x','y','z']]=after_vmax[['mvir','rvir','rs','rs_klypin','x','y','z']].div(h)
    
    # Sort by vmax
    after_vmax=after_vmax.sort_values(by=['vmax'],ascending=False)
    
    # Compute halo distances and create a new column
    # in comoving units
    after_vmax['dist']=np.linalg.norm(after_vmax[['x','y','z']]*1000-after_vmax.iloc[after_vmax.mvir.values.argmax()][['x','y','z']]*1000, None,1)

    # Indices have been shuffled when halos were sorted by Vmax.
    # reset_index() will assign new indices starting from 0 while keeping
    # the old (mixed) indices in column "index".
    after_vmax=after_vmax.reset_index()
    # Delete the 'index' column with old indices
    after_vmax=after_vmax.drop(columns=['index'])
    # Reset it once again - 'index' column holds the new indices
    after_vmax=after_vmax.reset_index()
    # Name this index column 'halo_id'
    # I don't think I will actually need this column.
    after_vmax=after_vmax.rename(columns={'index':'halo_id'})

    final_df=after_vmax[['host_id', 'halo_id', 'orig_id','mvir','vmax','rvir','rs','rs_klypin','x','y','z', 'vx', 'vy', 'vz','dist']]

    # Save the output file
    final_df.to_csv(out_full_path, index=False)
    
    #return(final_df)
    

'''
- Get halo particle IDs

- Takes in function parameters (catalog_path, snapshot_num, halo_id)
    - catalog_path: path to catalog directory
    - snapshot_num: snapshot number to use
    - halo_id: ID of the halos I want to use
        - For each .ascii file, it looks for the existence of the ID of interest,
          and the file that contains it will be read in to give hald id - num_p pairs.
  
  
- For now:
    - absolute paths will be used
    - think about more efficient ways
'''

def get_particle_id(catalog_path, snapshot_num, halo_id):
    # Get id - num_p pairs from the .ascii file
    id_num_p_pair, bin_file_name=get_halo_id_num_p(catalog_path, snapshot_num, halo_id)
    # print(id_num_p_pair[:3])
    # print(id_num_p_pair[-3:])
    
    # Number of halos is the length of the dataframe
    num_halos=len(id_num_p_pair)
    
    # Open the binary file
    bin_file=open(bin_file_name, 'rb')
    
    # Skip header and halo tables
    bin_file.seek(256+296*num_halos)
    
    # Get number of particles of each halo as an array
    num_p_array=id_num_p_pair.num_p.values

    # Get integer particle IDs and append the particle ID arrays (tuples) to a list
    s=time.time()
    particle_ID_list=[]
    for num_p in num_p_array:
        particle_IDs=struct.unpack("Q" * num_p, bin_file.read(8 * num_p))
        particle_ID_list.append(particle_IDs)
    
    #
    # Check if EOF has been reached: "not remaining_dat" is True if remaining_dat is empty.
    remaining_dat = bin_file.read()
    if not remaining_dat:
        # 
        print(f"  * Successfully reached EOF!")
    else:
        print(f"  * EOF not reached!")
        print(f"    * Remaining data is: {remaining_dat}")
    '''
    if bin_file.read() == None or ' ':
        print("Successfully reached EOF!")
        print()
        
    else:
        print("Not EOF!")
        print("Remaining data is:")
        print(bin_file.read())
        print()
    '''
        
    e=time.time()

    print("* get_particle_id() compute time: ", e-s, "seconds")
    print()
        
    return (particle_ID_list)
      

'''
- Get halo id (id) and number of particles (num_p) from the .ascii file

- file_name is the path to the directory .ascii files are stored in.
- Reading in .ascii file as a dataframe works far better than reading it in as a text file.
- Everything in this script comes from my script, make_csv_catalog.py.
- The first 19 rows contain simulation/snapshot information
- Rockstar-Galaxies Version: is the last line
- df.iloc[index] gives the row at index

- Returns:
  (1) binary file name created from the name of the ascii file used
  (2) data frame containing four columns ['index', 'old_index', 'id', 'num_p']
    - index: the first halo in the file obtains index 0, second halo 1 and so on
    - old_index: begins at 19 because halo table begins at line 19 in the original text file
    - id: halo ID from Rockstar output
    - num_p: number of particles in each halo

- For now:
    - absolute paths will be used
    - think about more efficient ways

- For later:
    - Think about changning the format of the input.
    - Instead of having the full path to the file as the function parameter,
      I can have snapshot numbers then create the file names using those.
    - I could even do this outside of this function so I can generate file names for
      .ascii, .bin, etc. files at the same time.
'''

def get_halo_id_num_p(catalog_path, snapshot_num, halo_id):
    # Create .ascii file names
    file_names=get_file_name(catalog_path, snapshot_num, 'ascii')
    
    for file_name in file_names:

        # Open file
        text_df=pd.read_csv(file_name, sep=" ", low_memory=False)
    
        # Take the data part
        reduced_data=text_df[19:]
    
        # I just need two columns: id and num_p
        two_columns=reduced_data[['#id', 'num_p']]
        two_columns=two_columns.rename(columns={'#id':'id'})
        two_columns=two_columns.astype(int)
    
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

'''
- Create full path names for Rockstar output files

- Input: (catalog_path, snapshot_num)
- Returns a list with 8 .ascii file names
- It is a very simple function but I will probably need it very often

- For later:
    - I could probably modify this slightly to make names for files in other formats
    - A more generic one?
'''

def get_file_name(catalog_path, snapshot_num, file_type='ascii'):
    file_base=f'{catalog_path}halos_{snapshot_num:03d}'
    file_name_list=[]
    
    # Create .ascii file names
    if file_type=='ascii':
        # Assuming 8 files
        for i in range(8):
            file_name=f'{file_base}.{i}.ascii'
            file_name_list.append(file_name)
            
    elif file_type=='bin':
        for i in range(8):
            file_name=f'{file_base}.{i}.bin'
            file_name_list.append(file_name)

    return(file_name_list)


'''
- Get the rvmax column from the .ascii file that contains the given halo

- Function is written in "tracked_particles_from_stampede.ipynb"
- Move it in here!
'''

