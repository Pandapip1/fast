# Author: Artur Meller <ameller@wustl.edu>, Maxwell I. Zimmerman <mizimmer@wustl.edu>
# Contributors:
# Copywright (C) 2017, Washington University in St. Louis
# All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited
# Proprietary and confidential


#######################################################################
# imports
#######################################################################


import gc
import glob
import itertools
import mdtraj as md
import numpy as np
import os
import sys
from .base_analysis import base_analysis
from .. import tools
import pickle


#######################################################################
# code
#######################################################################


def _calculate_number_of_contacts(traj, interface_list, dist_cutoff, 
                                  verbose=False):
    """Compute the number of contacts at the specified interfaces

    Parameters
    ----------
    traj : md.Trajectory
        The trajectory to do the computation for
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the residue indices (zero indexed) of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_rids, BH_rids], [BH_rids, FH_rids]]

    verbose : boolean
        Determines if intermediate scores are printed


    Returns
    -------
    contacts_per_interface : np.array, shape=(len(traj), len(interface_list))
        A contact count for each frame of `traj` and each interface
        in the interfacee list

    """
    contacts_per_interface = np.zeros((traj.n_frames, len(interface_list)), dtype=int)

    for i, (interface1_rids, interface2_rids) in enumerate(interface_list):
        contacts = np.zeros((traj.n_frames))
        # to avoid out of memory issues iterate over each residue at a time
        for j, rid in enumerate(interface2_rids):
            rid_pairs = list(itertools.product(interface1_rids, [rid]))

            # returns distances for each residue-residue contact in each frame of the trajectory
            # distances is shape=(n_frames, n_pairs)
            distances = md.compute_contacts(traj, rid_pairs, scheme='closest')[0]

            contacts = contacts + np.sum(distances < dist_cutoff, axis=1)

        contacts_per_interface[:, i] = contacts

    if verbose:
        print(contacts_per_interface)
        np.save('./data/contacts_per_interface.npy', contacts_per_interface)

    return contacts_per_interface.sum(axis=1)



class InterfaceContactWrap(base_analysis):
    """Computes the number of total residue contacts at multiple interfaces.

    Parameters
    ----------
    interface_list : list of of list of numpy arrays
        List containing the interfaces for which to compute scores
        Follows the pattern:
        [interface1, interface2, ...] where
        interfaceN is a 2-element list of numpy arrays containing
        the residue indices (zero indexed) of one half of the interface and the other half
        For example, the following would be a valid input:
        [[S2_rids, BH_rids], [BH_rids, FH_rids]]

    verbose : boolean
        Determines if intermediate scores are printed

    dist_cutoff : double
        Cutoff distance that defines what constitutes a contact. Any two residues
        that are less than the dist_cutoff apart are considered a contact.


    Attributes
    ----------
    output_name : str,
        The file containing rankings.
    """
    def __init__(
            self, interface_list, dist_cutoff, 
            verbose=False, base_struct=None):
        # load in interface list
        if type(interface_list) is str:
            with open(interface_list, 'rb') as f:
                self.interface_list = pickle.load(f)
        else:
            self.interface_list = interface_list
        self.dist_cutoff = dist_cutoff
        self.verbose = verbose

    @property
    def class_name(self):
        return "InterfaceContactWrapper"

    @property
    def config(self):
        return {
            'interface_list': self.interface_list,
        }

    @property
    def analysis_folder(self):
        return None

    @property
    def base_output_name(self):
        return "contacts_per_state"

    def run(self):
        # determine if file already exists
        if os.path.exists(self.output_name):
            pass
        else:
            # load centers
            centers = md.load(
                "./data/full_centers.xtc", top="./prot_masses.pdb")
            # calculate and save contacts
            scores = _calculate_number_of_contacts(centers,
                self.interface_list, self.dist_cutoff,
                verbose=self.verbose)
            np.save(self.output_name, scores)
