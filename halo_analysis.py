'''

'''
#-----------------------------------------------------------------------------------
# Import libraries.
#-----------------------------------------------------------------------------------
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import NFW
import h5py
#import matplotlib
#from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
#from scipy.signal import argrelextrema, find_peaks
from scipy.signal import argrelextrema, savgol_filter
from scipy.interpolate import interp1d
import sys
import struct
import time
#-----------------------------------------------------------------------------------
# Import local modules/libraries/scripts.
#-----------------------------------------------------------------------------------
import halo_utilities
import utilities
#-----------------------------------------------------------------------------------
# Classes
#-----------------------------------------------------------------------------------
'''
Halo dictionary-class properties:
    ***** Need to make sure if distance units are in over-h or non-h units!
    "ID.halo.infall": halo ID at the infall snapshot
        - This will often be used as the halo ID (hID).
        - There could be duplicate hIDs across snapshots.
        - But they are unique within a snapshot, so always use them together.
        - And use the treeID where available as it is unique within the simulation.
    "snapshot.infall": infall snapshot number
    "file.path": full directory/file name for the current halo's particle data
    "coordinates": 3-D positions of particles in simulation units [kpc comoving]
        - Each element contains 3-D coordinates for all of halo's particles within a single snapshot.
    "velocities": 3-D velocities of particles in simulation units [km/s]
        - Each element contains 3-D velocities for all of halo's particles within a single snapshot.
    "ID.particle": IDs of particles
    "snapshot.number": snapshot numbers from infall to the end of simulation
    
'''
class halo(dict):
    '''
    * A dictionary-class for handling the data for a single halo.
    '''
    def __init__(self, hID, infall_snapshot, fname, simulation_snapnum_dict, BH_parameters, out_f):  # , hID, infall_snap
        '''
        * Method to assign instance attributes.
        - ID.halo.infall: halo ID at the infall snapshot - this will be the halo ID.
          ***** REMEMBER *****
            - There could be duplicate ID.halo.infall values across snapshots.
            - But they are unique within a snapshot, so always use them together.
        - snapshot.infall: infall snapshot number
        - file.path: full directory/file name for the particle data.
        '''
        self["ID.halo.infall"] = hID
        self["snapshot.infall"] = infall_snapshot
        self["file.path"] = fname
        self["simulation.snapshot.number.dict"] = simulation_snapnum_dict
        self["h"] = BH_parameters['h']
        self["interpolation.factor"] = BH_parameters["interp.factor"]
        self["most.bound.fraction"] = BH_parameters['most_bound_frac']
        self["most.bound.number.min"] = BH_parameters['most_bound_min']
        self["most.bound.number.max"] = BH_parameters['most_bound_max']
        self["particle.mass"] = BH_parameters['part_mass'] / self["h"]# Assumes equal mass particles, so just one number, non-h M_solar
        self["cv.rapid.drop.fraction"] = BH_parameters['cv_rapid_drop_frac']
        self["cv.infall.drop.fraction"] = BH_parameters['cv_infall_frac']
        self["cv.stays.low.fraction"] = BH_parameters['cv_stays_low_frac']
        self["BH.parameters"] = BH_parameters # doing this removes the need to store all of the above parameters separately, but I am leaving it as it is right now.
        self["out_f"] = out_f # out_f is the text output file object, not just the name of it.
        # Initialize the halo data.
        # I could use individual methods here, but I decided to use a wrapper method "initialize_halo".
        self.initialize_halo()
    #
    def initialize_halo(self):
        '''
        * This function initializes the halo object by setting the particle data.
        '''
        # Set the halo particle data.
        self.set_particle_data()
        # Determine number of most bound particles to use.
        self.most_bound_particle_number()
        # Compute and store the center of mass of halo particles at each snapshot.
        # These initial COMs will be replaced later with more accurate ones.
        self.compute_com()
    #
    def set_particle_data(self):
        '''
        * Method to read in the particle data for the current halo and set the particle data as attributes for the current halo object.
        - utilities.open_halo_particles_file reads in the coordinates in over-h units.
        - So, convert it to non-h units here.
        - Coordinates and velocities will be stored in co-moving units.
        '''
        # Use function "open_halo_particles_file" from utilities.py to read in halo particles.
        # "open_halo_particles_file" needs the full file path.
        snapnums, coordinates, velocities, particleIDs = utilities.open_halo_particles_file(self["file.path"])
        # Assign the particle data as attributes.
        self["coordinates"] = coordinates / self["h"] # ckpc
        self["velocities"] = velocities # km/s / sqrt(a), co-moving
        self["ID.particle"] = particleIDs
        self["snapshot.number"] = snapnums
        self["number.of.particles.infall"] = len(self["ID.particle"])
        # Assign scale factors and redshifts corresponding to the snapshot numbers.
        intersecting_snapnums, intersecting_idx_sim, intersecting_idx_halo = np.intersect1d(self["simulation.snapshot.number.dict"]["snapshot_numbers"], self["snapshot.number"], assume_unique=True, return_indices=True)
        self["scale.factor"] = self["simulation.snapshot.number.dict"]["scale_factors"][intersecting_idx_sim]
        self["redshift"] = self["simulation.snapshot.number.dict"]["redshifts"][intersecting_idx_sim]
    #
    def most_bound_particle_number(self):
        '''
        * Method to determine the number of most bound particles to use.
        - Cases:
            1. num_total < most_bound_number_min: num_most_bound = num_total
            2. num_most_bound < most_bound_number_min: num_most_bound = most_bound_number_min
            3. most_bound_number_min <= num_most_bound <= most_bound_number_max: num_most_bound = num_part * most_bound_fraction
            4. num_most_bound > most_bound_number_max: num_most_bound = most_bound_number_max
        '''
        # Multiply the number of particles by the most bound fraction parameter.
        num_most_bound = int(self["number.of.particles.infall"] * self["most.bound.fraction"])
        if num_most_bound < self["most.bound.number.min"]:
            # When num_most_bound is smaller than the minimum most bound particle number limit:
            if self["number.of.particles.infall"] < self["most.bound.number.min"]:
                # If the total number of halo particles is smaller than the minimum most bound particle number limit, use all of the particles.
                num_most_bound = self["number.of.particles.infall"]
            else:
                # If the total number of halo particles is equal to or greater than the minimum most bound particle number limit, use the limit itself.
                num_most_bound = self["most.bound.number.min"]
        elif num_most_bound > self["most.bound.number.max"]:
            num_most_bound = self["most.bound.number.max"]
        # Set the computed number as an attribute of the halo object.
        self["most.bound.number"] = int(num_most_bound)
    #
    def compute_com(self):
        '''
        * Method to compute the center of mass of the particles recursively.
            - All snapshots
            - in comoving kpc
        '''
        com_list = []
        #
        for i in range(len(self["snapshot.number"])):
            # Compute the COM for each snapshot, using the cm function from halo_utilities.py.
            # I think: num_part is the minimum number of particles required to do the computation iteratively. If the halo has fewer particles than num_part, COM will be computed using all particles directly.
            # I think: nel_lim sets down to how many particles the iteration will be done.
            com = halo_utilities.cm(self["coordinates"][i], nofile=True, num_part=self["most.bound.number.min"], nel_lim=int(self["most.bound.number.min"] * 0.9), print_statement=False)
            # Append the result to the result list.
            com_list.append(com)
        # Set the center array as an attribute.
        self["com"] = np.array(com_list)
    #
    def compute_particle_energies_at_infall(self):
        '''
        * Method to compute kinetic, potential, and binding energies of halo particles at the halo's infall snapshot.
        - All computations are done in physical units.
        - Cosmological simulations using GIZMO use co-moving units.
        - To convert them to physical units:
            - length_physical = length_comoving * scale_factor
            - velocity_physical = velocity_comoving * sqrt(scale_factor)
        - Velocity units do not have a factor of h.

        - Particle energies at the infall snapshot are computed mainly to find the n% most bound particles at the infall snapshot. n will usually be small with the default value 2. For a massive halo, even 2% may be a large number and particle energy calculations for a large number of particles could be time consuming. So this method sets a minimum and maximum numbers of (inner-most) particles to compute the energies for.
        * Cases:
            - think about it!
        '''
        '''
compute_particle_energies_at_infall(current_halo.dmo_com[0], current_halo.dmo_coords[0], current_halo.dmo_vels[0],   current_halo.scale_factors[0], part_mass)
        '''
        scale_at_infall = self["scale.factor"][0]
        com_at_infall = self["com"][0] * scale_at_infall # non-h pkpc
        particle_coords_at_infall = self["coordinates"][0] * scale_at_infall # non-h pkpc
        particle_vels_at_infall = self["velocities"][0] * np.sqrt(scale_at_infall) # physical
        particle_mass = self["particle.mass"]
        # Number of particles to compute energies for.
        # Assuming most of the most bound particles are in the inner regions, use up to 10x more particles than the number of most bound particles used.
        # Default most.bound.number should be capped at 500, so the maximum value of n_use should be 5000.
        # n_use will be used to slice arrays that are sorted by the particle distance, so if the total number of particles is below n_use, the total number of particles will be used.
        n_use = int(self["most.bound.number"] * 10)
        # Use the function compute_particle_energies to compute the binding energies.
        _, _, self["particle.binding.energies.infall"], self["distance.sort.idx.infall"] = compute_particle_energies(com_at_infall, particle_coords_at_infall, particle_vels_at_infall, n_use, particle_mass)
    #
    def most_bound_particle_com(self):
        '''
        * This method uses the N% most bound particle subset at the infall snapshot to compute more accurate COMs for the halo at all following snapshots.
        - Note:
            - 
        '''
        # List to store the COM results.
        new_com_list = []
        # Particle coordinate data for all snapshots.
        coords_arr = self["coordinates"]
        # Argsort the infall snapshot binding energy array.
        bind_energy_sort_idx = np.argsort(self["particle.binding.energies.infall"])
        # Get N% most bound particles at infall. The exact number is already computed and stored as self["most.bound.number"].
        num_part = self["most.bound.number"]
        # Get indices of particles to use.
        most_bound_sort_idx = bind_energy_sort_idx[:num_part]
        track_idx = self["distance.sort.idx.infall"][most_bound_sort_idx]
        # Store the indices that give the most bound particles at infall.
        self["most.bound.indices"] = track_idx
        # Compute new COMs using most bound particles at infall.
        for i in range(len(coords_arr)):
            # Coordinates of the most bound particles at infall at the ith snapshot.
            current_snap_coords = coords_arr[i][track_idx]
            # Compute COM.
            com = halo_utilities.cm(current_snap_coords, nofile=True, num_part=self["most.bound.number.min"], nel_lim=int(self["most.bound.number.min"] * 0.9), print_statement=False)
            # Append the COM result to the result list.
            new_com_list.append(com)
        # Store the COM result.
        self["com.most.bound"] = np.array(new_com_list)
    #
    def compute_halo_properties(self, host_main_branch_dict):
        '''
        * Method to compute various halo properties.
        * Properties:
            Vmax -
            Rmax -
            cv -
            Rbound - 
            halo_velocity - 
            distance.from.host.ckpc - 
            scale.disrupt
            closest.approach.distance - 
              - if number.of.pericenters > 0:
                - min(pericenters)
              - if number.of.pericenters == 0:
                - min(distance)
            closest.approach.scale - 
            number.of.pericenters - pericenters exist only for those with a proper orbit. Subhalos disrupting before completing a (nearly) full orbit has number.of.pericenters = 0.
        '''
        scales = self["scale.factor"]
        # Create empty arrays to store results.
        #vcirc_results = np.zeros(len(scales), dtype=object)
        #sorted_pdist_results = np.zeros(len(scales), dtype=object)
        #density_results = np.zeros(len(scales), dtype=object)
        #bin_mid_results = np.zeros(len(scales), dtype=object)
        #rho_r_squared_results = np.zeros(len(scales), dtype=object)
        #local_min_idx_results_all = np.zeros(len(scales), dtype=object)
        #
        vmax_array = np.zeros(len(scales))
        rmax_array = np.zeros(len(scales))
        cv_array = np.zeros(len(scales))
        rboundary_array = np.zeros(len(scales))
        nboundary_array = np.zeros(len(scales), dtype=int)
        halo_velocity_list = []
        # Loop through each snapshot.
        for i in range(len(scales)):
            current_scale = scales[i]
            current_com = self["com.most.bound"][i] * current_scale # pkpc
            current_coords = self["coordinates"][i] * current_scale # pkpc
            current_velocities = self["velocities"][i] * np.sqrt(current_scale) # physical
            # Use the function vcirc_particle_single_halo() from halo_utilities.py to compute Vcirc: sorted_pdist_arr contains the distances of subhalo particles from the subhalo center, in physical kpc.
            vcirc_arr, sorted_pdist_arr, sort_idx = halo_utilities.vcirc_particle_single_halo(current_com, current_coords)
            # Use the function compute_density_profile_sorted_pdist() from halo_utilities.py to compute the density profile.
            # The first few bins often contain very few particles if the first bin starts from the first index, so start from a few indices back.
            num_part = len(sorted_pdist_arr)
            if num_part >= 100:
                first_idx = 10
            elif num_part < 100:
                first_idx = int(np.rint(0.1 * num_part))
            ### The last few particles often are outliers, so don't use them: this criterion has been removed.
            last_idx = -1
            density_arr, bin_mid_arr, num_enclosed_within_mid_arr = halo_utilities.compute_density_profile_sorted_pdist(sorted_pdist_arr, self["particle.mass"], first_idx, last_idx)
            # Compute rho(r) * r^2: this will be used to find the boundary of the halo.
            rho_r_squared = density_arr * bin_mid_arr * bin_mid_arr
            # Use the function concentration_one_halo() from halo_utilities.py to compute Vmax, Rmax, and cv, using the Vcirc profile just computed.
            ### R_boundary at infall: the distance that the density profile was computed out to. last_idx is currently set as -1. This part is removed.
            # r_boundary at infall: use the third to last particle's distance.
            if i == 0:
                r_boundary = sorted_pdist_arr[-3]
                # For the first infall snapshot, n_boundary uses all particles in sorted_pdist_arr.
                n_boundary = len(sorted_pdist_arr)
                #r_boundary = bin_mid_arr[-1]
                # Save some of the properties at infall.
                r_boundary_infall = r_boundary
                infall_min_density = np.min(density_arr)
                # At the infall snapshot, simply compute Vmax, rmax, and cv using all particles.
                cv, vmax, rmax = halo_utilities.concentration_one_halo(vcirc_arr, sorted_pdist_arr)
                # Compute the halo velocity using 10% inner-most particles.
                inner_most_frac = 0.1
                vels_sorted_by_distance = current_velocities[sort_idx]
                halo_velocity = compute_halo_velocity(inner_most_frac, vels_sorted_by_distance)
                #
                local_min_idx = [None]
            else:
                '''
                * Finding the turn-over point of the rho * r^2 profile to find the "edge" of the subhalo.
                '''
                r_boundary = find_r_boundary(density_arr, bin_mid_arr, sorted_pdist_arr, sort_idx, last_idx, r_boundary_infall, r_boundary_prev, infall_min_density, previous_min_density, current_scale, previous_scale, scales[0])
                # Compute Vmax, Rmax, and cv using only particles within r_boundary.
                slice_idx = np.searchsorted(sorted_pdist_arr, r_boundary)
                if slice_idx < 10:
                    # Anything that has a small slice_idx value probably are not to be trusted. But that "small" number also probably is much much larger than 10. But here, just use 10 so the code doesn't break.
                    # I think it will be really easy and obvious to tell which subhalos to remove from the analysis anyway.
                    slice_idx = 10
                cv, vmax, rmax = halo_utilities.concentration_one_halo(vcirc_arr[:slice_idx], sorted_pdist_arr[:slice_idx])
                # Compute the halo velocity using 10% inner-most particles.
                inner_most_frac = 0.1
                vels_sorted_by_distance = current_velocities[sort_idx[:slice_idx]]
                halo_velocity = compute_halo_velocity(inner_most_frac, vels_sorted_by_distance)
                # Number of particles within r_boundary
                n_boundary = slice_idx
            # Some of the properties will be used at the next snapshot.
            r_boundary_prev = r_boundary
            v_halo_prev = halo_velocity
            previous_scale = current_scale
            # minimum density of the subhalo WITHIN r_boundary.
            #min_density_idx = np.searchsorted(bin_mid_arr, r_boundary, side='right')
            previous_min_density = np.min(density_arr)
            # Add the result from the current scale factor to the result arrays.
            vmax_array[i] = vmax
            rmax_array[i] = rmax
            cv_array[i] = cv
            rboundary_array[i] = r_boundary
            nboundary_array[i] = n_boundary
            halo_velocity_list.append(halo_velocity)
        #
        halo_velocity_array = np.array(halo_velocity_list)
        # Compute halo's distance from the host halo at all snapshots.
        self['distance.from.host.ckpc'] = compute_distance_from_host(self["com.most.bound"], scales, host_main_branch_dict) # comoving, non-h kpc
        # Find the disruption scale factor.
        disruption_scale = find_disruption(cv_array, scales, self["cv.rapid.drop.fraction"], self["cv.infall.drop.fraction"], self["cv.stays.low.fraction"])
        # Compute halo's pericenter information: between infall and disruption.
        # Assume there are no broken-link subhalos and set infalling = 1. This might need to be revised if tracking for broken-link subhalos is to become available in later versions of Bloodhound.
        # Using the updated version of compute_pericenters: this needs the radial velocities of the subhalo with respect to the host halo.
        # Only use the subhalo data from the first (infall) snapshot to disruption snapshot.
        if disruption_scale == -1:
            last_idx = len(scales)
        else:
            last_idx = np.where(np.isclose(scales, disruption_scale, atol=1e-4))[0][0]+1
        halo_scales_first_to_last = scales[:last_idx]
        halo_coords_arr = self["com.most.bound"] * scales[:,None] # physical kpc
        halo_coords_first_to_last = halo_coords_arr[:last_idx] # physical kpc
        #halo_vels_first_to_last = halo_velocity_array[:last_idx]
        # Getting the slice of the host halo data that matches with the snapshots of the subhalo.
        start_scale = scales[0]
        host_first_idx = np.where(np.isclose(host_main_branch_dict['scale.factor'], start_scale, atol=1e-4))[0][0]
        #host_last_idx = host_first_idx+last_idx
        host_coords_arr = np.vstack((
            host_main_branch_dict['x'],
            host_main_branch_dict['y'],
            host_main_branch_dict['z']
        )).T[host_first_idx:] * scales[:, None] # physical kpc
        host_vels_arr = np.vstack((
            host_main_branch_dict['vx'],
            host_main_branch_dict['vy'],
            host_main_branch_dict['vz']
        )).T[host_first_idx:]
        host_coords_arr_first_to_last = host_coords_arr[:last_idx]
        #host_vels_arr_first_to_last = host_vels_arr[:last_idx]
        #
        if len(halo_scales_first_to_last) != len(host_coords_arr_first_to_last):
            print(f"  *** Error: halo_coords_arr and host_coords_arr have different lengths - {len(halo_coords_arr)}, {len(host_coords_arr_first_to_last)}, ID.halo.infall={self['ID.halo.infall']}, snapshot.infall={self['snapshot.infall']}", flush=True, file=self["out_f"])
        # Compute the radial velocities of the subhalo with respect to the host halo.
        halo_radial_vels_arr = get_radial_velocities(halo_coords_arr, halo_velocity_array, host_coords_arr, host_vels_arr)
        # Compute the pericenter information.
        pericenters, pericenter_scales, closest_peri_dist, closest_peri_scale = compute_pericenters(halo_coords_first_to_last, halo_radial_vels_arr[:last_idx], host_coords_arr_first_to_last, halo_scales_first_to_last, self["BH.parameters"], self["ID.halo.infall"], self["out_f"])# All physical units
        # Old pericenter function
        #pericenters, pericenter_scales, num_peri, closest_approach_dist, closest_approach_scale = compute_pericenters(self["com.most.bound"], scales, disruption_scale, host_main_branch_dict, self["interpolation.factor"], order=20, infalling=1)
        # Halo's relative 3D coordinates and velocities wrt the host, from infall to the end of the simulation
        halo_rel_coords_arr =  halo_coords_arr - host_coords_arr # physical kpc
        halo_rel_vels_arr = halo_velocity_array - host_vels_arr
        # Store results.
        self['host.coordinates'] = halo_rel_coords_arr # halo's 3D coordinates wrt the host
        self['vmax'] = vmax_array
        self['rmax'] = rmax_array
        self['scale.radius.klypin'] = rmax_array / 2.1626
        self['cv'] = cv_array
        self['radius.boundary'] = rboundary_array
        self['n.within.r.boundary'] = nboundary_array
        self['halo.velocity'] = halo_velocity_array
        self['host.velocity'] = halo_rel_vels_arr # halo's 3D velocities wrt the host
        self['host.velocity.rad'] = halo_radial_vels_arr # halo's radial velocity wrt the host
        self['scale.factor.disrupt'] = disruption_scale
        self['closest.pericenter'] = closest_peri_dist # physical, non-h kpc
        self['scale.factor.closest.pericenter'] = closest_peri_scale
        self['number.of.pericenters'] = len(pericenters)
    #
    def make_halo_tracking_dataframe(self):
        '''
        * This method takes an analyzed halo object and makes a (tree-like) halo tracking datatable with halo properties at tracked snapshots.
          - Returns a pandas dataframe.
          - Keys to skip:
            - file.path
            - simulation.snapshot.number.dict
            - h
            - most.bound.fraction
            - most.bound.number.min
            - most.bound.number.max
            - cv.rapid.drop.fraction
            - cv.infall.drop.fraction
            - cv.stays.low.fraction
            - coordinates
            - velocities
            - ID.particle
            - most.bound.number
            - particle.binding.energies.infall
            - distance.sort.idx.infall
            - most.bound.indices
            - com
          - Keys to use as it is:
            - snapshot.number
            - scale.factor
            - redshift
            - distance.from.host.ckpc
            - vmax
            - rmax
            - scale.radius.klypin
            - cv
            - radius.boundary
            - n.within.r.boundary
            - host.velocity.rad
          - Keys to be used with some modification:
            - ID.halo.infall - single number, make an array of the same number
            - snapshot.infall - single number, make an array of the same number
            - number.of.particles.infall ''
            - Not stored anymore: particle.mass - ''
            - scale.factor.disrupt - ''
            - closest.pericenter - ''
            - scale.factor.closest.pericenter - ''
            - number.of.pericenters - ''
            - com.most.bound - separate into x, y, z
            - host.coordinates - separate into host.x, host.y, host.z
            - halo.velocity - separate into vx, vy, vz
            - host.velocity - separate into host.vx, host.vy, host.vz
          - Skip unless in keys_use or keys_use_with_mod.
        '''
        # Data to use as it is.
        keys_use = ['snapshot.number', 'scale.factor', 'redshift', 'distance.from.host.ckpc', 'vmax', 'rmax', 'scale.radius.klypin', 'cv', 'radius.boundary', 'n.within.r.boundary', 'host.velocity.rad']
        keys_use_with_mod = ['ID.halo.infall', 'snapshot.infall', 'number.of.particles.infall', 'scale.factor.disrupt', 'com.most.bound', 'host.coordinates', 'closest.pericenter', 'scale.factor.closest.pericenter', 'number.of.pericenters', 'halo.velocity', 'host.velocity']
        # Data to use with some modification to the data structure.
        # An empty dictionary to append key-value pairs - will be converted to a dataframe at the end.
        result_dict = {}
        # Key names in the halo object.
        key_names = list(self.keys())
        # Length of the output arrays: equal to the number of snapshots the halo was tracked for.
        arr_len = len(self['scale.factor'])
        for i in range(len(key_names)):
            current_key = key_names[i]
            if current_key in keys_use:
                # No modification required: store the data in result_dict.
                result_dict[current_key] = self[current_key]
            elif current_key in keys_use_with_mod:
                if current_key == 'ID.halo.infall' or current_key == 'snapshot.infall' or current_key == 'number.of.particles.infall' or current_key == 'scale.factor.disrupt' or current_key == 'closest.pericenter' or current_key == 'scale.factor.closest.pericenter' or current_key == 'number.of.pericenters':
                    current_dat = self[current_key]
                    result_dict[current_key] = np.full(arr_len, current_dat)
                if current_key == 'com.most.bound':
                    # Separate the coordinates into x, y, z.
                    current_dat = self[current_key]
                    result_dict['x'] = current_dat[:,0]
                    result_dict['y'] = current_dat[:,1]
                    result_dict['z'] = current_dat[:,2]
                if current_key == 'host.coordinates':
                    # Separate the relative coordinates into host.x, host.y, host.z.
                    current_dat = self[current_key]
                    result_dict['host.x'] = current_dat[:,0]
                    result_dict['host.y'] = current_dat[:,1]
                    result_dict['host.z'] = current_dat[:,2]
                if current_key == 'halo.velocity':
                    # Separate the velocities into vx, vy, vz.
                    current_dat = self[current_key]
                    result_dict['vx'] = current_dat[:,0]
                    result_dict['vy'] = current_dat[:,1]
                    result_dict['vz'] = current_dat[:,2]
                if current_key == 'host.velocity':
                    # Separate the relative velocities into host.vx, host.vy, host.vz.
                    current_dat = self[current_key]
                    result_dict['host.vx'] = current_dat[:,0]
                    result_dict['host.vy'] = current_dat[:,1]
                    result_dict['host.vz'] = current_dat[:,2]
        # Convert the final dictionary into a Pandas dataframe.
        result_df = pd.DataFrame(index=None).from_dict(result_dict)
        # Store the result.
        self['tracking.df'] = result_df
#-----------------------------------------------------------------------------------
# Fuctions
#-----------------------------------------------------------------------------------
def analyze_halo(halo_ID, infall_snapshot, particle_fname, host_main_branch_dict, snapnum_info_dict, BH_parameters, out_f):
    '''
    * This is a wrapper function for performing subhalo analysis for ONE halo.
    - Files/data required:
      - host tree main branch file: DMO and Disk
      - DMO surviving halo main branch file
      - DMO subtree main branch file
      - Disk subtree file with subhalos found by using the infall criteria
    '''
    # Initialize and set halo data for the current subhalo.
    halo_obj = halo(halo_ID, infall_snapshot, particle_fname, snapnum_info_dict, BH_parameters, out_f)
    # Compute particle energies at the infall snapshot.
    halo_obj.compute_particle_energies_at_infall()
    # Compute more accurate COMs using most bound particles at infall.
    halo_obj.most_bound_particle_com()
    # Compute halo properties at each snapshot.
    halo_obj.compute_halo_properties(host_main_branch_dict)
    #
    # Create a "tree-like" subhalo tracking datatable for the current halo.
    halo_obj.make_halo_tracking_dataframe()
    #print(halo_obj["ID.halo.infall"], halo_obj["snapshot.infall"], flush=True, file=out_f)
    #print(halo_obj["file.path"], flush=True, file=out_f)
    #print(halo_obj["number.of.particles.infall"], flush=True, file=out_f)
    #print(halo_obj["most.bound.number"], flush=True, file=out_f)
    #print(halo_obj["com"][:3], flush=True, file=out_f)
    #print(halo_obj["com.most.bound"][:3], flush=True, file=out_f)
    #print("", flush=True, file=out_f)
    return(halo_obj)
#
def compute_distance_from_host(halo_com_arr, halo_scale_arr, host_main_branch_dict):
    host_scale_arr = host_main_branch_dict['scale.factor']
    #host_com_arr = np.vstack((host_main_branch_dict['x'], host_main_branch_dict['y'], host_main_branch_dict['z'])).T * 1000. 
    host_com_arr = np.vstack((host_main_branch_dict['x'], host_main_branch_dict['y'], host_main_branch_dict['z'])).T
    # Find, for host's arrays, the index that corresponds to the infall scale factor.
    start_scale = halo_scale_arr[0]
    host_first_idx = np.where(np.isclose(host_scale_arr, start_scale, atol=1e-4))[0][0]
    host_com_arr_use = host_com_arr[host_first_idx:]
    # Compute halo's distances.
    if len(halo_com_arr) != len(host_com_arr_use):
        print(f"*** compute_distance_from_host:")
        print(f"    - halo_com_arr and host_com_arr_use have different lengths: {len(halo_com_arr)}, {len(host_com_arr_use)}")
        print(f"    - start_scale = {start_scale}")
        print(f"    - host_first_idx = {host_first_idx}")
        print(f"    - host_scale at host_first_idx = {host_scale_arr[host_first_idx]}")
        print(f"    - halo_scale_arr: {halo_scale_arr}")
        print(f"    - host_scale_arr[host_first_idx:]: {host_scale_arr[host_first_idx:]}")
    halo_dist_comov = np.linalg.norm(halo_com_arr - host_com_arr_use, None, 1)
    #print(halo_dist_comov[-1])
    return(halo_dist_comov)
#
def compute_halo_velocity(inner_most_frac, sorted_vels):
    '''
    * Function to compute the (COM) velocity of a halo at a given snapshot.
    - It uses N% innermost particles and their velocities to compute the bulk velocity.
    - It assumes that the input velocity array, sorted_vels, is already sorted by particle distance.
    - Rockstar uses 10%.
    * Input:
    - inner_most_frac: fraction of the innermost particles to use (N / 100).
    - sorted_vels: particle velocity array sorted by particle distance.
    '''
    # Number of particles to compute the velocity with.
    tot_numpart = int(len(sorted_vels))
    numpart_use = int(tot_numpart * inner_most_frac)
    # Velocity array to use: the array is already assumed to be sorted.
    vels_use = sorted_vels[:numpart_use]
    # Compute the bulk velocity.
    halo_vel = halo_utilities.bulk_velocity(vels_use)
    # Return the result.
    return(halo_vel)
#
def compute_particle_energies(halo_com, particle_coords, particle_vels, n_use, particle_mass):
    '''
    * Function to compute the kinetic, gravitational potential, and binding energies for a given set of particles.
    - I assume everything is already in physical units, but it doesn't matter as long as everything is consistent.
    - Use 10% inner-most particles to compute the halo velocity, then compute the kinetic energies of particles: similar to how Rockstar does it.
    - Returns:
        - kinetic energy array
        - potential energy array,
        - binding energy array,
        - index array that would sort the distance array.
    '''
    # Convert coordinates into halocentric units: with respect to the center of the halo.
    rel_coords = particle_coords - halo_com
    # Compute particles' distance and get the sorting indices.
    dist_arr = np.linalg.norm(rel_coords, None, 1)
    dist_sort_idx_arr = np.argsort(dist_arr) # This takes ~0.1s for 1,000,000 elements on my laptop, so it should not cause issues.
    # Distance, coordinates, and velocity arrays sorted by particle distance.
    sorted_dist_arr = dist_arr[dist_sort_idx_arr]
    sorted_coords = rel_coords[dist_sort_idx_arr]
    sorted_vels = particle_vels[dist_sort_idx_arr]
    # Compute the halo velocity.
    inner_most_frac = 0.1
    halo_velocity = compute_halo_velocity(inner_most_frac, sorted_vels)
    # Convert sorted particle velocities into halocentric units: with respect to the halo velocity.
    rel_sorted_vels = sorted_vels - halo_velocity
    # Compute kinetic energies: square the velocity magnitude.
    kin_energy_arr = 0.5 * particle_mass * np.sum((rel_sorted_vels[:n_use] * rel_sorted_vels[:n_use]), axis=1)
    # Compute potential energies.
    # For now, compute the potential energy of each particle due to ALL other halo particles, directly: might want to implement Barnes-Hut or something.
    pot_energy_arr = halo_utilities.particle_PE_direct_sum(sorted_coords[:n_use], sorted_coords, particle_mass)
    # Compute binding energies: It's actually the total energy instead of the negative of it, so the most bound particle has the smallest/most negative value.
    bind_energy_arr = kin_energy_arr + pot_energy_arr
    # Return results.
    return(kin_energy_arr, pot_energy_arr, bind_energy_arr, dist_sort_idx_arr)
#
def find_disruption(cv_array, scale_array, cv_rapid_drop_fraction, cv_infall_drop_fraction, cv_stays_low_fraction):
    '''
    * This function determines when the given subhalo disrupts.
    * How I determine disruption:
      1) First, find snapshots where cv decreases to below 30% of the previous snapshot.
      2) Of these, check if cv is below 20% of cv at infall:
          - Often, a subhalo's cv increases a little when the subhalo crosses the host boundary. It might make more sense use that value than the cv value at infall as a reference point?
      3) Also check if cv does not back back to above 40% of cv at infall in the next 5 snapshots.
    *** Disruption time is defined as the snapshot immediately before the subhalo is found to be completely
        disrupted: so it's the last snapshot the subhalo is found to be a halo.
    * Input:
      - cv_array: cv values for all snapshots
      - scale_array: scale factor array
    * Output:
      - 
    * Things to consider, for the future:
      - Should I add a minimum cv value requirement as well?
        - E.g. cv below 1000 or something like that
      - Using the maximum cv value within the first 3-5 snapshots as the "infall" reference value?
    '''
    '''
    * Rapid drop criteria:
    '''
    # Cv at infall
    cv_infall = cv_array[0]
    #cv_infall = np.max(cv_array[:5])
    #print(cv_infall, np.max(cv_array[:5]))
    # Cv normalized by Cv at infall: infall is the first element of the array.
    cv_norm = cv_array / cv_infall
    # Compute how much the value of cv changes by at each snapshot.
    cv_change = cv_norm[1:] / cv_norm[:-1]
    # Find where cv_change decreases down to the amount set by cv_rapid_drop_fraction.
    cv_rapid_drop_idx = np.where(cv_change <= cv_rapid_drop_fraction)[0]
    #
    # disrupt_found checks if an instance that meets all disruption criteria is found.
    disrupt_found = False
    #
    # Check each of the instances in cv_rapid_drop_idx to identify the disruption snapshot.
    for i in range(len(cv_rapid_drop_idx)):
        idx = cv_rapid_drop_idx[i]
        #
        # Of these, find those that have cv below cv_infall_drop_fraction of cv at infall.
        # Check that cv doesn't come back up to cv_stays_low_fraction of cv at infall in the next 5 snapshots.
        # idx + 1 because cv_change is computed from the second element of cv_norm.
        # The first instance that satisfies all three criteria is what we are looking for.
        if cv_norm[idx+1] < cv_infall_drop_fraction and np.max(cv_norm[idx+1:idx+12]) < cv_stays_low_fraction:
            # The current cv rapid drop instance also satisfies the other two disruption criteria: significant cv reduction compared to cv at infall, and stays reduced for at least 5 snapshots.
            # Disruption time is defined as the snapshot immediately before the subhalo is found to be disrupted: so it's the last snapshot the subhalo is found to be a halo.
            # So use idx instead of idx+1.
            disrupt_scale = scale_array[idx]
            disrupt_found = True
        #
        # If disruption is found, return result: returning terminates the loop.
        if disrupt_found:
            return(disrupt_scale)
    # If it loops all the way to the end, the halo does not disrupt: return -1.
    return(-1)
#
def find_r_boundary(density_arr, bin_mid_arr, sorted_pdist_arr, sort_idx, last_idx, r_boundary_infall, r_boundary_prev, infall_min_density, previous_min_density, current_scale, previous_scale, infall_scale):
    '''
    * This function finds the turn-over point (r_boundary) in the density profile and separates the "true" density profile part from the added contribution from stripped particles.
    * rho(r)*r^2 turn-over point criteria:
      - First, use scipy's argrelextrema to find local minima, then use the following criteria to find a turn-over point.
        - not the first nor last element of the array
        - smaller than N% of the first value of rho? rho*r^2?
        - located within 300% of r_boundary_infall (comparison must be done in comoving units!)
        - located between 40-200% of r_boundary_prev (comparison must be done in comoving units!)
        - then choose the one with the smallest density value.
      - If there are there are no local minima, or none of the minima meet the criteria:
        - look for zero values of rho*r^2
        - places where the density is below N% of the smallest value of density at infall (comparison must be done in comoving units!): 
          - this aims to take care of cases where there is no clear turn-over point,
          - but the density gradually decreases to an 'unphysical' values.
          - E.g. N_p_infall=400, r_boundary_infall=10kpc subhalo having the density profile extend out to 50kpc surely has the real boundary well within 50kpc.
        - Figuring out a place where there is a big jump in the distance of particles in a sorted array of particles could also be a useful check for r_boundary!
    * r_boundary is important for two reasons:
      - 1. this will basically be used as the radius of the halo,
      - 2. halo properties will be computed only within r_boundary.
    '''
    # Convert physical units to comoving units.
    r_boundary_infall_comoving = r_boundary_infall / infall_scale
    r_boundary_prev_comoving = r_boundary_prev / previous_scale
    infall_min_density_comoving = infall_min_density * infall_scale * infall_scale * infall_scale
    previous_min_density_comoving = previous_min_density * previous_scale * previous_scale * previous_scale
    density_arr_comoving = density_arr * current_scale * current_scale * current_scale
    # Compute rho(r) * r^2: this will be used to find the boundary of the halo.
    rho_r_squared = density_arr * bin_mid_arr * bin_mid_arr
    first_rho_r_squared = rho_r_squared[0]
    last_rho_r_squared = rho_r_squared[-1]
    last_rho_r_squared_idx = int(len(rho_r_squared)-1)
    # Find local minima of the rho(r)*r^2 profile.
    local_min_idx = argrelextrema(rho_r_squared, np.less)
    # List for local minimum indices that meet the criteria: for one snapshot.
    min_idx_use = []
    # r_boundary_found tracks whether a density turn-over point (or a reasonable halo boundary) has been found or not.
    r_boundary_found = False
    # Check if local minima exist.
    if len(local_min_idx) > 0:
        # Local minima exist, so check if they meet the density turn-over criteria.
        for j in range(len(local_min_idx[0])):
            # Local minimum index to use
            current_min_idx = local_min_idx[0][j]
            # Value of rho and rho*r^2 at the current index
            current_min_rho = density_arr_comoving[current_min_idx]
            current_min_rho_r_squared = rho_r_squared[current_min_idx]
            # Value of distance at the current index, in comoving units
            current_min_r_comoving = bin_mid_arr[current_min_idx] / current_scale
            # Check if the current minimum satisfies the criteria.
            #if (current_min_idx!=0) and (current_min_idx!=last_rho_r_squared_idx) and \
            #(current_min_rho < 0.05*density_arr_comoving[0]) and \
            #(current_min_r_comoving < 3.*r_boundary_infall_comoving) and \
            #(0.4*r_boundary_prev_comoving < current_min_r_comoving < 2.*r_boundary_prev_comoving):
            if (current_min_idx!=0) and (current_min_idx!=last_rho_r_squared_idx) and \
            (current_min_rho_r_squared < 0.02*first_rho_r_squared) and \
            (current_min_r_comoving < 3.*r_boundary_infall_comoving) and \
            (0.4*r_boundary_prev_comoving < current_min_r_comoving < 2.*r_boundary_prev_comoving):
                min_idx_use.append(current_min_idx)
    if len(min_idx_use) == 1:
        # There is exactly one minimum that meets all criteria: use this as the turn-over point.
        turn_over_idx = min_idx_use[0]
        r_boundary = bin_mid_arr[turn_over_idx]
        r_boundary_found = True
        test = 'minimum found'
    elif len(min_idx_use) > 1:
        # There are multiple minima that meet all criteria.
        # Choose the first occurance of these minima as the true minimum.
        turn_over_idx = min_idx_use[0]
        r_boundary = bin_mid_arr[turn_over_idx]
        test = 'minimum found'
        r_boundary_found = True
    if r_boundary_found == False:
        # No suitable r_boundary found from the local minimum test.
        # This automatically takes care of both len(min_idx_use)==0 and len(local_min_idx)==0.
        # Do the next r_boundary test: look for density bins with zero values.
        zero_value_bin_idx = np.where(rho_r_squared == 0.)[0]
        zero_idx_use = []
        if len(zero_value_bin_idx) > 0:
            # There are bins with a zero density value.
            # Check if these look like the "edge" of the halo.
            for j in range(len(zero_value_bin_idx)):
                current_zero_idx = zero_value_bin_idx[j]
                current_zero_r_comoving = bin_mid_arr[current_zero_idx] / current_scale
                # Check if these zero value points satisfy the distance criteria.
                if (current_zero_r_comoving < 3.*r_boundary_infall_comoving) and \
                (0.4*r_boundary_prev_comoving < current_zero_r_comoving < 2.*r_boundary_prev_comoving):
                    zero_idx_use.append(current_zero_idx)
            if len(zero_idx_use) > 0:
                # At least one reasonable zero-value point is found: use the first occurance of these as the turn-over point.
                turn_over_idx = zero_idx_use[0]
                r_boundary = bin_mid_arr[turn_over_idx]
                r_boundary_found = True
                test = 'r_boundary found from zero density value check'
    if r_boundary_found == False:
        # No suitable r_boundary found from zero density bin check.
        # Do the next r_boundary test: look for a place in the sorted particle distance array where the difference between adjacent particles is over 1kpc.
        pdist_difference_arr = sorted_pdist_arr[1:] - sorted_pdist_arr[:-1]
        pdist_big_jump_idx = np.where(pdist_difference_arr>1.)[0]
        big_jump_idx_use = []
        if len(pdist_big_jump_idx) > 0:
            # There are big jumps in particle distances.
            # Check if these are located in a reasonable place that makes it a reasonable "edge" of the halo.
            for j in range(len(pdist_big_jump_idx)):
                current_big_jump_idx = pdist_big_jump_idx[j]
                # pdist_at_idx_comoving would be r_boundary if it satisfies the distance criteria.
                pdist_at_idx_comoving = sorted_pdist_arr[current_big_jump_idx] / current_scale
                if (pdist_at_idx_comoving < 3.*r_boundary_infall_comoving) and \
                (0.4*r_boundary_prev_comoving < pdist_at_idx_comoving < 2.*r_boundary_prev_comoving):
                    big_jump_idx_use.append(current_big_jump_idx)
            if len(big_jump_idx_use) > 0:
                # At least one reasonable particle distance jump is identified: use the first occurance as the turn-over point (r_boundary).
                r_boundary = sorted_pdist_arr[big_jump_idx_use[0]]
                r_boundary_found = True
                test = 'r_boundary found from particle distance jump check'
    if r_boundary_found == False:
        # No suitable r_boundary found from particle distance jump check.
        # Do the next r_boundary test: look for a place where the density is below N% of the smallest value at infall or N'% of the smallest value at the previous snapshot.
        #small_density_idx = np.where((density_arr_comoving < 0.5*infall_min_density_comoving) | (density_arr_comoving < 0.7*previous_min_density_comoving))[0]
        small_density_idx = np.where((density_arr_comoving < 0.5*infall_min_density_comoving))[0]
        if len(small_density_idx) > 0:
            # Unusually small density values are identified.
            # Check if any of these meet the criteria.
            r_at_small_density_comoving = bin_mid_arr[small_density_idx] / current_scale
            small_density_idx_use = np.where((r_at_small_density_comoving < 3.*r_boundary_infall_comoving) & (r_at_small_density_comoving > 0.4*r_boundary_prev_comoving) & (r_at_small_density_comoving < 2.*r_boundary_prev_comoving))[0]
            if len(small_density_idx_use) > 0:
                # There are density values that satisfy the small density criteria.
                # Use the first occurance as the "turn-over" point.
                small_density_idx_use = small_density_idx_use[0]
                r_boundary = bin_mid_arr[small_density_idx[small_density_idx_use]]
                r_boundary_found = True
                test = 'r_boundary found from small density check'
    if r_boundary_found == False:
        # Absolutely no reasonable density turn-over point or halo boundary found.
        r_boundary = sorted_pdist_arr[-1]
        test = 'no r_boundary found: using the farthese particle distance as r_boundary'
    return(r_boundary)
#
def get_physical_x_y_z(coords_arr, scale_arr):
    return(coords_arr * scale_arr[:,None])
#
def distance_interpolation_arr(dist_arr, scale_arr, interp_factor):
    npts = int(len(dist_arr) * interp_factor)
    # Interpolate the distance/radius array.
    dist_interp = interp1d(scale_arr, dist_arr, kind='linear')
    new_scale_arr = np.linspace(scale_arr[0], scale_arr[-1], npts)
    # New distance/radius array.
    new_dist_arr = dist_interp(new_scale_arr)
    return(new_dist_arr, new_scale_arr)
#
def distance_interpolation_coords(sub_coords, host_coords, scale_arr, interp_factor):
    # Compute dx, dy, dz: component-wise distance from the center of the host halo.
    dcoords = host_coords - sub_coords
    npts = int(len(dcoords) * interp_factor)
    #
    # Interpolate dx, dy, dz arrays as a function of the scale factor.
    x_interp = interp1d(scale_arr, dcoords[:,0], kind='linear')
    y_interp = interp1d(scale_arr, dcoords[:,1], kind='linear')
    z_interp = interp1d(scale_arr, dcoords[:,2], kind='linear')
    #
    # A new scale factor array with more data points:
    new_scale_arr = np.linspace(scale_arr[0], scale_arr[-1], npts)
    #
    # A new distance array using the interpolated function:
    new_x = x_interp(new_scale_arr)
    new_y = y_interp(new_scale_arr)
    new_z = z_interp(new_scale_arr)
    new_dist_arr = (new_x**2. + new_y**2. + new_z**2.)**0.5
    return(new_dist_arr, new_scale_arr) 
#
def get_radial_velocities_tree(tree_df, tree_coords_arr, host_coords_arr, host_velocity_arr):
    '''
    * Note: I find that host.v* are different from computing v* - host's v*. So host.velocity.rad and computed v_rad are also different.
            For the cases I checked, this also changes the location of where the radial velocity changes sign...
            In most cases, I expect 'host.v*' and 'host.velocity.rad' to exist in the dataset, so let's hope that these are accurate...
    '''
    # tree_coords_arr and host_coords_arr are in physical kpc.
    # Check if the merger tree data already contains halo's radial velocity wrt the host halo.
    colmap = {c.lower(): c for c in tree_df.columns} # lowercase -> original column name map
    # Case 1: radial velocity exists in the tree data.
    if 'host.velocity.rad' in colmap:
        # Radial velocity data already exists: use it.
        return(tree_df[colmap['host.velocity.rad']].values)
    # Case 2: Radial velocity doesn't exist, but relative velocities exist.
    if all(k in colmap for k in ['host.vx', 'host.vy', 'host.vz']):
        v_rel = tree_df[[colmap['host.vx'], colmap['host.vy'], colmap['host.vz']]].values
    else:
        # Use vx, vy, vz to compute the radial velocity.
        sub_vels = tree_df[['vx', 'vy', 'vz']].values # 3D velocities wrt the simulation box
        v_rel = sub_vels - host_velocity_arr
    # Relative position vector
    r_vec = tree_coords_arr - host_coords_arr
    r_mag = np.linalg.norm(r_vec, axis=1)
    # Avoid division by zero
    r_hat = np.zeros_like(r_vec)
    valid_idxs = r_mag > 0
    r_hat[valid_idxs] = r_vec[valid_idxs] / r_mag[valid_idxs, None]
    v_rad = np.sum(v_rel * r_hat, axis=1)
    return(v_rad)
#
def get_radial_velocities(halo_coords_arr, halo_vels_arr, host_coords_arr, host_vels_arr):
    '''
    * This function computes the radial velocities of the subhalo with respect to the host halo using the 3D velocities of the subhalo and host halo.
    * Note: relative velocities between the host and subhalo must be computed first because there's no host.vx etc. available here.
    * halo_coords_arr and host_coords_arr are assumed to be in physical kpc.
    '''
    # Compute the relative 3D velocities between the host and subhalo.
    v_rel = halo_vels_arr - host_vels_arr
    # Relative position vector
    r_rel = halo_coords_arr - host_coords_arr
    r_mag = np.linalg.norm(r_rel, axis=1)
    # Compute the radial velocities: use np.clip to avoid division by zero.
    v_rad = np.sum(r_rel * v_rel, axis=1) / np.clip(r_mag, 1e-8, None)
    return(v_rad)
#
def find_first_infall_from_tree(halo_tree_df, host_scale_arr, host_radius_arr):
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
    # Scale factor and distance for the subtree: flip the order to early to late times.
    # Host's data is already flipped.
    sub_scale_arr = halo_tree_df['scale.factor'].values[::-1]
    sub_dist_arr = halo_tree_df['distance.from.host.ckpc'].values[::-1]
    #
    # Find indices to slice the data for the host: idx_2 handles subtrees.
    idx_1 = np.where(np.isclose(host_scale_arr, sub_scale_arr[0], atol=1e-04))[0][0]
    idx_2 = np.where(np.isclose(host_scale_arr, sub_scale_arr[-1], atol=1e-04))[0][0]
    # Host's radii corresponding to the scale factors for the subtree.
    host_radius_slice = host_radius_arr[idx_1:idx_2+1]
    # Compare subhalo's distance to host's rvir and find first infall.
    dist_subtracted = sub_dist_arr - host_radius_slice
    #
    # Find the first infall: first occurence of a negative value in the subtracted array.
    first_infall_idx = np.where(dist_subtracted < 0)[0]
    if len(first_infall_idx) > 0:
        # Subhalo has infall(s).
        if first_infall_idx[0] > 0:
            # If the first time the subhalo is found inside the Rvir is not the first index,
            # set the snapshot immediately before as the first infall snapshot and return its scale factor.
            first_infall_scale = sub_scale_arr[first_infall_idx[0]-1]
            infall_idx = first_infall_idx[0] - 1
            infalling = 1
            #
            return(first_infall_scale, infall_idx, infalling)
        else:
            # Index is zero: the subtree is formed inside the Rvir of the host halo.
            # This is not an "infalling" subhalo, but keep its data.
            first_infall_scale = sub_scale_arr[first_infall_idx[0]]
            infall_idx = first_infall_idx[0]
            infalling = 0
            return(first_infall_scale, infall_idx, infalling)
            #return(-1., -1, infalling)
        '''
                elif first_infall_idx[0] == 0 and sub_scale[0] <= 0.25:
            first_infall_scale = first_infall_idx[0]
            infall_idx = first_infall_idx[0]
            return(first_infall_scale, infall_idx)
        '''
    else:
        # Subhalo never falls within rvir of the host.
        return(-1., -1, 0)
#
def compute_pericenters(sub_coords, sub_radial_vels, host_coords, scale_arr, parameter_dict, sub_tid, out_f):
    '''
    * This function computes the pericenter information for a subhalo given the coordinates of the subhalo and the host halo, from the infall snapshot to the last snapshot.
    * It uses scipy's interp1d to interpolate the coordinate arrays between snapshots.
    * In the future, I will update the interpolation routine to use the orbital dynamics rather than simply interpolating between numbers.
    * Cases:
      - Subhalo has fewer than four snapshots: don't interpolate, num_peri=1, closest_peri=min(distance)
      - Subhalo has infalling==0: subhalo starts within the host halo, treat the first snapshot as an additional pericenter if the distance is smaller than the next snapshot.
        - This has been dropped.
      - Subhalo has no extreme points: num_peri=1, closest_peri=min(interpolated distance)
      - Subhalo has extreme points: num_peri=len(pericenters), closest_peri=min(pericenters)
    * Use Savitzky-Golay Filter (scipy.signal.savgol_filter) to smooth the distance array.
      - To remove "wiggles",
      - Use the smoothed distance array to find rough locations of pericenters (argrelextrema),
      - Then refine the pericenter by finding the true minimum in the raw array nearby - within some window.
      - savgol_filter can shift the exact index of the pericenter if the distance elements around the pericenter is not symmetric (most cases are not symmetric).
      - savgol_filter window_length: size of the window to smooth over. Use a fairly small number since I just want to smooth over little wiggles without smoothing over an actual orbit.
      - For my analyses, I will use int(300 Myr / snapshot spacing): typical orbital time will be > 1 Gyr, but using 300 Myr to include extreme cases.
        - 300 Myr corresponds to a circular orbit with, Vcirc=200km/s, R=10kpc: 2 * pi * R / Vcirc = 300 Myr
        - More radial orbits with the same pericenter distance will have longer orbital periods.
        - So 300 Myr probably is a safe value.
      - For 600 snapshot FIRE runs, this will be around 13.
    * Input data:
      - sub_coords: x, y, z coordinates of the subhalo
      - sub_radial_vels: radial velocity of the subhalo wrt the center of the host halo: host.velocity.rad
    * Input parameters:
      parameter_dict = {"peri.upper.lim":150., # Not used anymore
              "interp.factor":interp_factor,
              "min.snaps.for.interp": 10,
              "savgol.window.size": savgol_window_target,
              "savgol.polyorder": 3,
              "peri.refine.half.width":21,
              "peri.vrad.buffer": interp.factor,
              "vrad.sign.frac.threshold":0.9
              }
    '''
    # Some validity checks for the input parameters
    if parameter_dict["peri.refine.half.width"] < 2:
        msg = "peri.refine.half.width must be >=2 for velocity checks."
        print(f"  * {msg}", flush=True, file=out_f)
        raise ValueError(msg)
    if parameter_dict["peri.vrad.buffer"] >= parameter_dict["peri.refine.half.width"]:
        msg = "peri.vrad.buffer must be < peri.refine.half.width"
        print(f"  * {msg}", flush=True, file=out_f)
        raise ValueError(msg)
    # --------------------------------------------
    # Case 1: enough snapshots to interpolate
    # --------------------------------------------
    if len(sub_coords) >= parameter_dict['min.snaps.for.interp']:
        # Subhalo has more than the minimum number of snapshots set by min_num_snaps.
        # Get interpolated distance and scale factor arrays.
        new_dist_arr, new_scale_arr = distance_interpolation_coords(sub_coords, host_coords, scale_arr, parameter_dict['interp.factor'])
        #new_host_radius_arr, new_scale_arr_1= distance_interpolation_arr(host_radius_arr, scale_arr, parameter_dict['interp.factor'])
        # Interpolate radial velocity onto same grid.
        new_vrad_arr, _ = distance_interpolation_arr(sub_radial_vels, scale_arr, parameter_dict['interp.factor'])
        # Smooth the distance array to identify approximate minima.
        smooth_dist_arr = savgol_filter(new_dist_arr, window_length=parameter_dict['savgol.window.size'], polyorder=parameter_dict['savgol.polyorder'])
        #
        # Find pericenter index candidates from the smoothed distance array: order sets how many points on each side to use for the comparison.
        dist_min_idx_candidates = argrelextrema(smooth_dist_arr, np.less, order=parameter_dict["peri.refine.half.width"])[0]
        # For each of the index candiates, search elements around it in the "raw" distance data (new_dist_arr) to find the true minimum.
        dist_min_idx_use = []
        for i in range(len(dist_min_idx_candidates)):
            # -------------------------------
            # Refine using RAW distance
            # -------------------------------
            idx_smooth = dist_min_idx_candidates[i]
            idx_low = max(0, idx_smooth - parameter_dict["peri.refine.half.width"])
            idx_high = min(len(new_dist_arr), idx_smooth+parameter_dict["peri.refine.half.width"]+1)
            idx_raw = idx_low + int(np.argmin(new_dist_arr[idx_low:idx_high]))
            # Check if the minimum is within the host's radius.
            ##### Host radius is currently not used for pericenter rejection.
            #idx_host_radius = new_host_radius_arr[idx_raw] # Host halo's radius at the current pericenter index.
            #idx_distance = new_dist_arr[idx_raw] # subhalo's distance at the current pericenter index.
            # ---------------------------------------------------------------------------------------------
            # Radial velocity sign-change check
            # Use buffer to avoid single-snapshot noise: host.v* are often different from those computed from [vx, vy, vz] - host's [vx, vy, vz]
            # and this can shift the exact location of where the sign of the radial velocity changes by a small number of snapshots (interpolated snapshots).
            # Adding a buffer: infalling -> transitioning (buffer) -> receding phases. 
            # ---------------------------------------------------------------------------------------------
            idx_start = max(0, idx_raw - parameter_dict["peri.refine.half.width"])
            idx_end = min(len(new_vrad_arr), idx_raw + parameter_dict["peri.refine.half.width"] + 1)
            vrad_before = new_vrad_arr[idx_start : idx_raw-parameter_dict["peri.vrad.buffer"]]
            vrad_after = new_vrad_arr[idx_raw+parameter_dict["peri.vrad.buffer"]+1 : idx_end]
            # Require sufficient samples
            do_vrad_check = True
            if len(vrad_before) < 3 or len(vrad_after) < 3:
                do_vrad_check=False
            if do_vrad_check:
                # Check the sign-consistency of the radial velocity before and after the pericenter.
                frac_infall = np.mean(vrad_before < 0.)
                frac_recede = np.mean(vrad_after > 0.)
                #print(frac_infall, frac_recede)
                if (frac_infall < parameter_dict["vrad.sign.frac.threshold"]) or (frac_recede < parameter_dict["vrad.sign.frac.threshold"]):
                    # Radial velocities before and after the pericenter do not have consistent signs.
                    continue
            # Passed all applicable checks.
            dist_min_idx_use.append(idx_raw)
        # Ensure at least one pericenter exists by definition:
        # the absolute minimum distance is always a pericenter
        abs_minimum_idx = np.argmin(new_dist_arr)
        dist_min_idx_use.append(abs_minimum_idx)
        dist_min_idx_use = np.unique(dist_min_idx_use) # remove duplicates, order by time, and convert to an array.
    # --------------------------------------------
    # Case 2: too few snapshots
    # --------------------------------------------
    else:
        # There are too few snapshots, so scipy interp1d will not work / or there's no point in interpolating / or there really is not an "orbit".
        dcoords = host_coords - sub_coords
        new_dist_arr = (dcoords[:,0]**2. + dcoords[:,1]**2. + dcoords[:,2]**2.)**0.5
        new_scale_arr = scale_arr
        dist_min_idx_use = np.array([np.argmin(new_dist_arr)])
    #
    # Get pericenters and pericenter scale factors.
    if len(dist_min_idx_use) > 0:
        pericenters = new_dist_arr[dist_min_idx_use]
        pericenter_scales = new_scale_arr[dist_min_idx_use]
    else:
        # There is no extreme point found: this should never happen!
        msg = f"*** {sub_tid}: Something went wrong: no pericenter found!"
        print(f"  * {msg}", flush=True, file=out_f)
        raise ValueError(msg)
    # Now that pericenters are found, get the closest pericenter information.
    if len(pericenters) > 0:
        #
        # Pericenters are found so choose the smallest pericenter as the closest pericenter.
        closest_peri_idx = np.argmin(pericenters)
        closest_peri_dist = pericenters[closest_peri_idx]
        closest_peri_scale = pericenter_scales[closest_peri_idx]
    else:
        # There is no pericenter found, so choose the smallest distance as the closest pericenter.
        # This case should never exist right?
        msg = f"*** {sub_tid}: Something went wrong: no pericenter found!"
        print(f"  * {msg}", flush=True, file=out_f)
        raise ValueError(msg)
    return(pericenters, pericenter_scales, closest_peri_dist, closest_peri_scale)
#
#def compute_pericenters(halo_coord_arr, halo_scale_arr, disruption_scale, host_main_branch_dict, interp_factor, order, infalling):
    '''
    ####### This function is replaced by a better one #######
    * This function computes the pericenter information of a given halo.
    * All distance values returned are in the same units (comoving, non-h kpc) as the input values.
    * Input:
      - halo_coord_arr: [x, y, z] coordinates of the subhalo, comoving non-h kpc
      - halo_scale_arr: scale factors corresponding to halo_coord_arr, from early to late times.
      - disruption scale: scale factor at the disruption time of the subhalo
      - host_main_branch_dict: dictionary containing the host main branch data
      - interp_factor: interpolation factor to achieve the target time spacing between interpolated distance elements. I have set the default target to be 0.001 in scale factor. For now, this is fixed and minimum is 2.
      - order: the order argument for scipy.signal.argrelextrema. This sets how many points on each side to use for the comparison. I use a fairly large number, 20, to help prevent mistaking small distance "wiggles" as relative extrema. With the time spacing of 0.001 in scale factor, order=20 corresponds to 0.02 in scale factor, which is a little over 300 Myr on average (scale factor and time do not have a linear relationship).
      - infalling: whether the subhalo has an infall or not. 1=infalling, 2=not infalling (i.e., subhalo "forms" within the host and is probably a broken-link subhalo). Here, every subhalo is assumed to have an infall.
    * Cases:
      - Subhalo has fewer than four snapshots: don't interpolate, num_peri=1, closest_peri=min(distance)
      - Subhalo has infalling==0: subhalo starts within the host halo, treat the first snapshot as an additional pericenter if the distance is smaller than the next snapshot.
      - Subhalo has no extreme points: num_peri=1, closest_peri=min(interpolated distance)
      - Subhalo has extreme points: num_peri=len(pericenters), closest_peri=min(pericenters)
    '''
    '''
    # Get host halo data.
    host_scale_arr = host_main_branch_dict['scale']
    host_coord_arr = np.vstack((host_main_branch_dict['x'], host_main_branch_dict['y'], host_main_branch_dict['z'])).T # Comoving, non-h kpc
    # Take the subhalo data between infall (first element) to the last snapshot (disruption time if disrupting, a=1 if surviving).
    '''
    '''
    if disruption_scale == -1:
        last_idx = None
    else:
        last_idx = np.where(np.isclose(halo_scale_arr, disruption_scale, atol=1e-4))[0][0]+1
    halo_coord_use = halo_coord_arr[:last_idx]
    halo_scale_use = halo_scale_arr[:last_idx]
    # Take host halo's coordinates from subhalo's infall snapshot to the last snapshot.
    first_idx = np.where(np.isclose(host_scale_arr, halo_scale_arr[0]))[0][0]
    host_coord_use = host_coord_arr[first_idx:first_idx+last_idx] # last_idx is shifted by first_idx.
    '''
    '''
    # Take the subhalo and host data between infall (first element) to the last snapshot (disruption time if disrupting, a=1 if surviving).
    # Left index to truncate the host halo's coordinate array with.
    first_idx = np.where(np.isclose(host_scale_arr, halo_scale_arr[0]))[0][0]
    if disruption_scale == -1:
        halo_coord_use = halo_coord_arr[:]
        halo_scale_use = halo_scale_arr[:]
        host_coord_use = host_coord_arr[first_idx:]
    else:
        last_idx = np.where(np.isclose(halo_scale_arr, disruption_scale, atol=1e-4))[0][0]+1
        halo_coord_use = halo_coord_arr[:last_idx]
        halo_scale_use = halo_scale_arr[:last_idx]
        host_coord_use = host_coord_arr[first_idx:first_idx+last_idx] # last_idx is shifted by first_idx.
    if len(halo_coord_use) > 3:
        # Subhalo data has tracking data for four or more snapshots.
        # Interpolate distance and scale factor arrays.
        new_dist_arr, new_scale_arr = halo_utilities.distance_interpolation(halo_coord_use, host_coord_use, halo_scale_use, interp_factor)
        # Find relative minima of the distance array.
        dist_minimum_idx = argrelextrema(new_dist_arr, np.less, order=order)[0]
        # Find peaks and filter the relative minima values.
        inverted_dist_arr = -new_dist_arr
        peak_idx, peak_properties = find_peaks(inverted_dist_arr, prominence=10.)
        # Adjust the prominence for the last N=order elements:
        # this is to handle the case where the last pericenter is less than prominence=10 kpc lower than any values between it and the last distance element due to the simulation ending while the subhalo is still near the pericenter.
        last_N_elements_peak_idx, last_N_elements_peak_properties = find_peaks(inverted_dist_arr[-order:], prominence=5.)
        last_N_elements_peak_idx = last_N_elements_peak_idx + len(inverted_dist_arr) - order
        # Combine the peak indices.
        peak_idx_combined = np.concatenate([peak_idx, last_N_elements_peak_idx])
        # Update the minima indices: take the overlap between dist_minimum_idx and peak_idx_combined.
        dist_minimum_idx_use = np.intersect1d(dist_minimum_idx, peak_idx_combined)
    else:
        # There are fewer than four snapshots, so scipy interp1d will not work.
        dcoord = host_coord_use - halo_coord_use
        new_dist_arr = (dcoord[:,0]**2. + dcoord[:,1]**2. + dcoord[:,2]**2.)**0.5
        new_scale_arr = halo_scale_use
        #dist_minimum_idx_use = np.array([np.argmin(new_dist_arr)])
        # Assume there are no pericenters when there are 3 or fewer snapshots.
        dist_minimum_idx_use = np.array([])
    if infalling == 0:
        # Subhalo starts within the host halo: such cases should not exist, but this part is included for completeness' sake.
        if new_dist_arr[0] < new_dist_arr[1]:
            # The distance at the first snapshot is smaller than that at the second snapshot, so the first one is a pericenter.
            dist_minimum_idx_use = np.insert(dist_minimum_idx_use, 0, 0)
    # Get pericenters and pericenter scale factors.
    if len(dist_minimum_idx_use) > 0:
        # Pericenters are found.
        pericenters = new_dist_arr[dist_minimum_idx_use]
        pericenter_scales = new_scale_arr[dist_minimum_idx_use]
        #pericenters_phys = pericenters * pericenter_scales
        num_peri = len(pericenters)
        closest_approach_idx = np.argmin(pericenters)
        closest_approach_dist = pericenters[closest_approach_idx]
        closest_approach_scale = pericenter_scales[closest_approach_idx]
    else:
        # No pericenters are found.
        pericenters = -1
        #pericenters_phys = -1
        pericenter_scales = -1
        num_peri = 0
        closest_approach_idx = np.argmin(new_dist_arr)
        closest_approach_dist = new_dist_arr[closest_approach_idx]
        closest_approach_scale = new_scale_arr[closest_approach_idx]
    return(pericenters, pericenter_scales, num_peri, closest_approach_dist, closest_approach_scale)
    '''
        