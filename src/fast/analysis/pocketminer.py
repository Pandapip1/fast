# SPDX-License-Identifier: LGPL-2.1-or-later
# Author: Gavin N. John <gjohn@caltech.edu>
# Contributors:
# Copyright (C) 2025, University of Pennsylvania

import numpy as np
import os
from fast.analysis.base_analysis import base_analysis

from fast.pocketminer.validate_performance_on_xtals import (
    process_strucs,
    predict_on_xtals,
)

from validate_performance_on_xtals import process_strucs, predict_on_xtals
import tempfile

import numpy as np
from scipy.spatial import Delaunay

import numpy as np
from scipy.spatial import Voronoi, ConvexHull


def make_predictions(strucs, model, nn_path):
    """
    strucs : list of single frame MDTraj trajectories
    model : MQAModel corresponding to network in nn_path
    nn_path : path to checkpoint files
    """
    X, S, mask = process_strucs(strucs)
    predictions = predict_on_xtals(model, nn_path, X, S, mask)
    return predictions


# From http://proteinsandproteomics.org/content/free/tables_1/table08.pdf
# In angstrom cubed
van_der_waals = {
    "A": 67,
    "R": 148,
    "N": 96,
    "D": 91,
    "C": 86,
    "E": 114,
    "Q": 109,
    "G": 48,
    "H": 118,
    "I": 124,
    "L": 124,
    "K": 135,
    "M": 124,
    "F": 135,
    "P": 90,
    "S": 73,
    "T": 93,
    "W": 163,
    "Y": 141,
    "V": 105,
}


def intersect_polyhedron_with_sphere(poly_points, center, radius):
    """
    Approximate the volume of the intersection between a polyhedron
    (defined by poly_points) and a sphere centered at 'center' with 'radius'.

    Parameters
    ----------
    poly_points : np.ndarray, shape=(M, 3)
        Vertices of the polyhedral region.
    center : np.ndarray, shape=(3,)
        Center of the bounding sphere.
    radius : float
        Radius of the sphere.

    Returns
    -------
    volume : float
        Approximate intersection volume.
    """
    # This requires mesh intersection; placeholder strategy:
    # Clip vertices to inside sphere, then take ConvexHull
    dists = np.linalg.norm(poly_points - center, axis=1)
    clipped = poly_points[dists <= radius]

    if len(clipped) >= 4:
        try:
            ch = ConvexHull(clipped)
            return ch.volume
        except:
            return 0.0
    else:
        return 0.0


def physically_capped_voronoi_score(points, probs, radii, bond_length=2.8):
    """
    Computes a geometry-aware score using Voronoi cells, capping infinite regions
    by ~~intersecting~~ replacing them with a sphere around the residue.

    Parameters
    ----------
    points : (N, 3) np.ndarray
        3D coordinates of residues.
    probs : (N,) np.ndarray
        Per-residue probabilities.
    radii : (N,) np.ndarray
        Known radii for each residue.
    bond_length : float
        Bond extension length (e.g., hydrogen bond reach).

    Returns
    -------
    score : float
        Weighted sum of (possibly capped) Voronoi volumes.
    """
    N = len(points)
    assert points.shape == (N, 3)
    assert probs.shape == (N,)
    assert radii.shape == (N,)

    try:
        vor = Voronoi(points)
    except:
        return np.sum(probs)  # fallback

    volumes = np.zeros(N)

    for i in range(N):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        center = points[i]
        cap_radius = radii[i] + bond_length

        if not region or -1 in region:
            # Infinite region: assume a spherical residue
            # (Not a joke)
            cap_radius = radii[i] + bond_length
            volumes[i] = (4 / 3) * np.pi * (cap_radius**3)
        else:
            try:
                verts = vor.vertices[region]
                ch = ConvexHull(verts)
                volumes[i] = ch.volume
            except:
                volumes[i] = 0

    return np.sum(volumes * probs)


class PocketMinerLikelihood(base_analysis):
    """
    Runs PocketMiner on a trajectory or list of PDBs and returns
    the maximum per-residue likelihood for each structure.

    Parameters
    ----------
    traj_path : str
        Path to MDTraj-compatible trajectory file (e.g., xtc).
    top_path : str
        Path to topology file (e.g., PDB).
    nn_path : str
        Path to PocketMiner neural network checkpoint.
    """

    def __init__(self, nn_path):
        self.nn_path = nn_path

        # Define MQA model parameters
        # create a MQA model
        DROPOUT_RATE = 0.1
        NUM_LAYERS = 4
        HIDDEN_DIM = 100

        # MQA Model used for selected NN network
        self.model = MQAModel(
            node_features=(8, 50),
            edge_features=(1, 32),
            hidden_dim=(16, HIDDEN_DIM),
            num_layers=NUM_LAYERS,
            dropout=DROPOUT_RATE,
        )

    @property
    def class_name(self):
        return "PocketMinerLikelihood"

    @property
    def config(self):
        return {
            "traj_path": self.traj_path,
            "top_path": self.top_path,
            "nn_path": self.nn_path,
            "invert": self.invert,
        }

    @property
    def base_output_name(self):
        return "pocketminer_likelihood"

    def run(self):
        if os.path.exists(self.output_name):
            return

        # load centers
        traj = md.load("./data/full_centers.xtc", top="./prot_masses.pdb")

        # Calculate scores
        scores = np.zeros(traj.n_frames)
        for i in range(traj.n_frames):
            preds = make_predictions([traj[i]], self.model, self.nn_path)
            scores[i] = physically_capped_voronoi_score(
                traj[i].xyz, preds, np.zeros(traj[i].xyz.shape[0])
            )

        np.save(self.output_name, scores)
