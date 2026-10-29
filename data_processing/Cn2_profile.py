"""
AOF Cn2 profiles (8 SPARTA layers with sample-dependent altitudes, placeholders and invalid entries) -> the fixed layers of the PSF model:

    repair     invalid layers are fixed inside each profile (an altitude placeholder gets the median altitude of its layer index,
               an invalid fraction becomes an empty layer), the fractions are renormalized; profiles without usable mass stay NaN
               and are imputed later from the other telemetry
    grid       weights of the N_layers PSF-model layers at fixed altitudes (the nodes, the same for all samples): every AOF layer splits
               its mass between the two nodes around it so that the total mass and the (5/3)-moment of the altitude, as the
               anisoplanatism scales, are preserved. The same weights are the Cn2 profile of the configs and the calibrator features

The grid (nodes, median profile) is fitted by STD_dataset_utils.store_Cn2_grid (nodes: the dataset medians of the per-sample K-means
altitudes; working copy in the STD folder) and copied into the distributed calibrator bundle (data_processing/NFM_bundle.py), the dataset
cache and the trained bundles, so that training and inference use the same reduction.
"""
import json
import numpy as np

N_AOF_LAYERS = 8
ALT_COLS     = [f'ALT{i}'          for i in range(1, N_AOF_LAYERS+1)] # [km]
FRAC_COLS    = [f'CN2_FRAC_ALT{i}' for i in range(1, N_AOF_LAYERS+1)]

MAX_ALTITUDE_KM  = 100.0 # above: placeholder (1000 km) or garbage
MIN_PROFILE_MASS = 0.5   # profiles with less valid mass are treated as missing
MOMENT_POWER     = 5/3


def grid_feature_names(N_layers):
    return [f'Cn2_grid_{k+1}' for k in range(N_layers)]


class Cn2Grid:
    def __init__(self, nodes_km, layer_altitudes_km, median_fracs=None, pin_ground_layer=True):
        self.nodes    = np.asarray(nodes_km, dtype=float).ravel() # [N_layers] altitudes of the PSF-model layers, increasing, nodes[0] = 0 (ground)
        self.N_layers = len(self.nodes)
        self.pin_ground_layer = bool(pin_ground_layer) # the first AOF layer (the profiler's ground layer, < 1.6 km) goes entirely to the ground node

        if self.N_layers < 2 or self.nodes[0] != 0.0 or np.any(np.diff(self.nodes) <= 0):
            raise ValueError('Grid nodes must start at 0 km and increase, with at least 2 nodes')

        self.median_fracs = np.full(self.N_layers, 1.0/self.N_layers) if median_fracs is None else np.asarray(median_fracs, dtype=float).ravel()
        self.median_fracs = self.median_fracs / self.median_fracs.sum()

        # Altitudes assigned to the AOF layers whose altitude is invalid (median altitude per layer index)
        self.layer_altitudes = np.asarray(layer_altitudes_km, dtype=float).ravel()
        if len(self.layer_altitudes) != N_AOF_LAYERS:
            raise ValueError(f'Expected {N_AOF_LAYERS} layer altitudes, got {len(self.layer_altitudes)}')

        self.feature_names = grid_feature_names(self.N_layers)

    # ------------------------------------------------- Serialization -------------------------------------------------
    def state_dict(self):
        return {
            'nodes_km':           self.nodes.tolist(),
            'layer_altitudes_km': self.layer_altitudes.tolist(),
            'median_fracs':       self.median_fracs.tolist(),
            'pin_ground_layer':   self.pin_ground_layer,
        }


    @classmethod
    def from_state_dict(cls, state):
        if 'boundaries_km' in state: # before 2026-10-29: 8 nodes binned into the model layers with per-sample heights
            raise ValueError('Legacy Cn2 grid (8 nodes + layer boundaries): rebuild it with STD_dataset_utils.store_Cn2_grid (create_STD_dataset.py)')
        return cls(**state)


    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.state_dict(), f, indent=2)


    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls.from_state_dict(json.load(f))


    def __repr__(self):
        return f'Cn2Grid(layers at {np.round(self.nodes, 2).tolist()} km)'


    # ------------------------------------------------- Profiles --------------------------------------------------------
    def repair(self, alts, fracs):
        """
        Fixes the invalid entries of the raw AOF profiles [N, 8] (altitudes in [km], fractions):
          - an altitude outside [0, 100] km, or 0 km for the layers above the first one (placeholder), is replaced by the median
            altitude of that layer index. The layers are not sorted: every layer is projected independently later
          - a fraction outside [0, 1] (NaN, 1e21 garbage) empties the layer
          - the fractions are renormalized to 1; profiles with less than MIN_PROFILE_MASS valid mass are returned as NaN
        Returns (alts, fracs, n_repaired [N], valid [N]); n_repaired counts the layers with mass whose altitude was replaced.
        """
        alts  = np.array(alts,  dtype=float, ndmin=2).copy()
        fracs = np.array(fracs, dtype=float, ndmin=2).copy()

        alt_ok = np.isfinite(alts) & (alts >= 0.0) & (alts < MAX_ALTITUDE_KM)
        alt_ok[:, 1:] &= alts[:, 1:] > 0.0
        frac_ok = np.isfinite(fracs) & (fracs >= 0.0) & (fracs <= 1.0)

        fracs[~frac_ok] = 0.0
        n_repaired = ((fracs > 1e-3) & ~alt_ok).sum(axis=1)
        alts = np.where(alt_ok, alts, self.layer_altitudes[None, :])

        mass  = fracs.sum(axis=1)
        valid = mass >= MIN_PROFILE_MASS
        fracs = np.where(valid[:, None], fracs / np.where(valid, mass, 1.0)[:, None], np.nan)
        alts  = np.where(valid[:, None], alts, np.nan)

        return alts, fracs, n_repaired, valid


    def to_grid(self, alts, fracs):
        """
        Repaired profiles [N, 8] -> weights of the fixed layers [N, N_layers]: the mass of a layer at altitude h between the nodes
        h_k <= h < h_k+1 goes to the two nodes with the weights (h_k+1^p - h^p) and (h^p - h_k^p), p = 5/3 (mass and (5/3)-moment
        preserved); layers above the last node go to it. With pin_ground_layer, the first AOF layer goes entirely to the ground node
        (it is the ground layer by construction and the only moving layer of the PSF model). NaN profiles stay NaN.
        """
        alts  = np.array(alts,  dtype=float, ndmin=2)
        fracs = np.array(fracs, dtype=float, ndmin=2)

        if self.pin_ground_layer:
            alts = alts.copy()
            alts[:, 0] = 0.0

        N     = alts.shape[0]
        grid  = np.zeros((N, self.N_layers))
        valid = np.isfinite(fracs).all(axis=1)

        h = np.clip(alts, self.nodes[0], self.nodes[-1])
        k = np.clip(np.searchsorted(self.nodes, h, side='right') - 1, 0, self.N_layers - 2) # lower node of every layer
        p = self.nodes**MOMENT_POWER
        t = (h**MOMENT_POWER - p[k]) / (p[k+1] - p[k]) # 0 at the lower node, 1 at the upper one
        t = np.where(np.isfinite(t), t, 0.0)
        w = np.where(valid[:, None], fracs, 0.0)
        rows = np.repeat(np.arange(N), alts.shape[1])
        np.add.at(grid, (rows, k.ravel()),   ((1.0 - t) * w).ravel())
        np.add.at(grid, (rows, k.ravel()+1), (t * w).ravel())
        grid[~valid] = np.nan

        return grid


    def process(self, alts, fracs):
        """Raw profiles [N, 8] -> (weights of the fixed layers [N, N_layers], NaN for missing profiles, n_repaired [N], valid [N])."""
        alts, fracs, n_repaired, valid = self.repair(alts, fracs)
        return self.to_grid(alts, fracs), n_repaired, valid


    def fraction_below(self, alts, fracs, h_km):
        """Fraction of the (repaired) profile mass below the altitude h_km, NaN for missing profiles."""
        alts  = np.array(alts,  dtype=float, ndmin=2)
        fracs = np.array(fracs, dtype=float, ndmin=2)
        below = np.where(np.isfinite(alts) & (alts < h_km), fracs, 0.0).sum(axis=1)
        return np.where(np.isfinite(fracs).all(axis=1), below, np.nan)


def fit_Cn2_grid(alts, fracs, nodes_km, layer_altitudes_km=None, pin_ground_layer=True):
    """
    Grid with the layers at nodes_km (first at 0 km) from a set of raw AOF profiles [N, 8]: the repair altitudes are the median altitudes
    of the layer indices, the median profile is the normalized median of the reprojected profiles.
    """
    alts  = np.array(alts,  dtype=float, ndmin=2)
    fracs = np.array(fracs, dtype=float, ndmin=2)
    alt_ok = np.isfinite(alts) & (alts >= 0.0) & (alts < MAX_ALTITUDE_KM) & np.isfinite(fracs) & (fracs > 0.0) & (fracs <= 1.0)
    alt_ok[:, 1:] &= alts[:, 1:] > 0.0

    layer_alts = np.array([np.median(alts[:, i][alt_ok[:, i]]) for i in range(N_AOF_LAYERS)]) if layer_altitudes_km is None \
                 else np.asarray(layer_altitudes_km, dtype=float)

    grid = Cn2Grid(nodes_km, layer_alts, pin_ground_layer=pin_ground_layer)
    weights, _, valid = grid.process(alts, fracs)
    median = np.median(weights[valid], axis=0)

    return Cn2Grid(nodes_km, layer_alts, median_fracs=median / median.sum(), pin_ground_layer=pin_ground_layer)
