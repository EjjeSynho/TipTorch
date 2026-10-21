"""
AOF Cn2 profiles (8 SPARTA layers with sample-dependent altitudes, placeholders and invalid entries) -> fixed representations:

    repair     invalid layers are fixed inside each profile (an altitude placeholder gets the median altitude of its layer index,
               an invalid fraction becomes an empty layer), the fractions are renormalized; profiles without usable mass stay NaN
               and are imputed later from the other telemetry
    grid       fractions at N_grid fixed altitude nodes (the median AOF layer altitudes): every layer splits its mass linearly between
               the two nodes around it, so the total mass and the mean altitude are preserved. Fixed-length calibrator input
    layers     the N_layers layers of the PSF model: fixed altitude boundaries between the nodes, the weights are the bin masses and
               the heights the (5/3)-moment effective altitudes of the bins (ground layer at 0), as the anisoplanatism scales

The grid (nodes, boundaries, median profile) is stored in data/reduced_telemetry/MUSE/AOF_Cn2_grid.json (STD_dataset_utils.store_Cn2_grid)
and copied into the dataset cache and the calibrator bundles, so that training and inference use the same reduction.
"""
import json
import numpy as np

N_AOF_LAYERS = 8
ALT_COLS     = [f'ALT{i}'          for i in range(1, N_AOF_LAYERS+1)] # [km]
FRAC_COLS    = [f'CN2_FRAC_ALT{i}' for i in range(1, N_AOF_LAYERS+1)]

MAX_ALTITUDE_KM  = 100.0 # above: placeholder (1000 km) or garbage
MIN_PROFILE_MASS = 0.5   # profiles with less valid mass are treated as missing
MOMENT_POWER     = 5/3


def grid_feature_names(N_grid):
    return [f'Cn2_grid_{k+1}' for k in range(N_grid)]


class Cn2Grid:
    def __init__(self, nodes_km, boundaries_km, median_fracs=None, layer_altitudes_km=None, pin_ground_layer=True):
        self.nodes      = np.asarray(nodes_km,      dtype=float).ravel() # [N_grid], increasing, nodes[0] = 0 (ground)
        self.boundaries = np.asarray(boundaries_km, dtype=float).ravel() # [N_layers-1], between the nodes
        self.N_grid     = len(self.nodes)
        self.N_layers   = len(self.boundaries) + 1
        self.pin_ground_layer = bool(pin_ground_layer) # the first AOF layer (the profiler's ground layer, < 1.6 km) goes entirely to the ground node

        if self.nodes[0] != 0.0 or np.any(np.diff(self.nodes) <= 0):
            raise ValueError('Grid nodes must start at 0 km and increase')
        if np.any(np.diff(self.boundaries) <= 0):
            raise ValueError('Layer boundaries must increase')

        self.median_fracs = np.full(self.N_grid, 1.0/self.N_grid) if median_fracs is None else np.asarray(median_fracs, dtype=float).ravel()
        self.median_fracs = self.median_fracs / self.median_fracs.sum()
       
        # Altitudes assigned to the AOF layers whose altitude is invalid (median altitude per layer index)
        self.layer_altitudes = self.nodes.copy() if layer_altitudes_km is None else np.asarray(layer_altitudes_km, dtype=float).ravel()
        if len(self.layer_altitudes) != N_AOF_LAYERS:
            raise ValueError(f'Expected {N_AOF_LAYERS} layer altitudes, got {len(self.layer_altitudes)}')

        self.layer_of_node = np.searchsorted(self.boundaries, self.nodes, side='right') # [N_grid] model layer of every node
        self.membership    = np.zeros((self.N_grid, self.N_layers)) # [N_grid, N_layers], 0/1
        self.membership[np.arange(self.N_grid), self.layer_of_node] = 1.0
        self.feature_names = grid_feature_names(self.N_grid)

    # ------------------------------------------------- Serialization -------------------------------------------------
    def state_dict(self):
        return {
            'nodes_km':           self.nodes.tolist(),
            'boundaries_km':      self.boundaries.tolist(),
            'median_fracs':       self.median_fracs.tolist(),
            'layer_altitudes_km': self.layer_altitudes.tolist(),
            'pin_ground_layer':   self.pin_ground_layer,
        }


    @classmethod
    def from_state_dict(cls, state):
        return cls(**state)


    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.state_dict(), f, indent=2)


    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            return cls.from_state_dict(json.load(f))


    def __repr__(self):
        return f'Cn2Grid(nodes {np.round(self.nodes, 2).tolist()} km, boundaries {self.boundaries.tolist()} km)'


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
        Repaired profiles [N, 8] -> fractions at the grid nodes [N, N_grid]: the mass of a layer at altitude h between the nodes
        h_k <= h < h_k+1 goes to the two nodes with the weights (h_k+1 - h) and (h - h_k) (mass and mean altitude preserved);
        layers above the last node go to it. With pin_ground_layer, the first AOF layer goes entirely to the ground node (it is
        the ground layer by construction and the only moving layer of the PSF model). NaN profiles stay NaN.
        """
        alts  = np.array(alts,  dtype=float, ndmin=2)
        fracs = np.array(fracs, dtype=float, ndmin=2)
        
        if self.pin_ground_layer:
            alts = alts.copy()
            alts[:, 0] = 0.0
            
        N     = alts.shape[0]
        grid  = np.zeros((N, self.N_grid))
        valid = np.isfinite(fracs).all(axis=1)

        h = np.clip(alts, self.nodes[0], self.nodes[-1])
        k = np.clip(np.searchsorted(self.nodes, h, side='right') - 1, 0, self.N_grid - 2) # lower node of every layer
        t = (h - self.nodes[k]) / (self.nodes[k+1] - self.nodes[k])                        # 0 at the lower node, 1 at the upper one
        t = np.where(np.isfinite(t), t, 0.0)
        w = np.where(valid[:, None], fracs, 0.0)
        rows = np.repeat(np.arange(N), alts.shape[1])
        np.add.at(grid, (rows, k.ravel()),   ((1.0 - t) * w).ravel())
        np.add.at(grid, (rows, k.ravel()+1), (t * w).ravel())
        grid[~valid] = np.nan
        
        return grid


    def to_layers(self, grid):
        """
        Grid fractions [N, N_grid] -> (weights [N, N_layers], heights [N, N_layers] in [km]) of the model layers: the weights are the
        masses of the altitude bins, the heights the (5/3)-moment effective altitudes of the bins (ground layer at 0 km). Empty bins
        get the heights of the median profile, NaN rows the median layers.
        """
        grid    = np.array(grid, dtype=float, ndmin=2)
        valid   = np.isfinite(grid).all(axis=1)
        g       = np.where(valid[:, None], grid, self.median_fracs[None, :])
        g       = g / np.clip(g.sum(axis=1, keepdims=True), 1e-12, None)
        weights = g @ self.membership
        moment  = (g * self.nodes[None, :]**MOMENT_POWER) @ self.membership
        heights = (moment / np.clip(weights, 1e-12, None))**(1.0/MOMENT_POWER)
        heights = np.where(weights > 1e-6, heights, self._median_heights[None, :])
        heights[:, 0] = 0.0
        
        return weights, heights


    @property
    def _median_heights(self):
        g      = self.median_fracs
        w      = g @ self.membership
        moment = (g * self.nodes**MOMENT_POWER) @ self.membership
        h      = (moment / np.clip(w, 1e-12, None))**(1.0/MOMENT_POWER)
        h[0]   = 0.0
        return h


    def default_layers(self):
        """(weights, heights [km]) of the median profile, used when a sample has no usable Cn2 profile."""
        w, h = self.to_layers(self.median_fracs[None, :])
        return w[0], h[0]


    def process(self, alts, fracs):
        """Raw profiles [N, 8] -> (grid [N, N_grid], weights [N, N_layers], heights [N, N_layers] km, n_repaired [N], valid [N])."""
        alts, fracs, n_repaired, valid = self.repair(alts, fracs)
        grid = self.to_grid(alts, fracs)
        weights, heights = self.to_layers(grid)
        return grid, weights, heights, n_repaired, valid


    def process_df(self, df):
        """The same from a DataFrame with the ALT{i} / CN2_FRAC_ALT{i} columns (missing columns count as missing profiles)."""
        alts  = np.column_stack([df[c].to_numpy(float) if c in df else np.full(len(df), np.nan) for c in ALT_COLS])
        fracs = np.column_stack([df[c].to_numpy(float) if c in df else np.full(len(df), np.nan) for c in FRAC_COLS])
        return self.process(alts, fracs)


    def fraction_below(self, alts, fracs, h_km):
        """Fraction of the (repaired) profile mass below the altitude h_km, NaN for missing profiles."""
        alts  = np.array(alts,  dtype=float, ndmin=2)
        fracs = np.array(fracs, dtype=float, ndmin=2)
        below = np.where(np.isfinite(alts) & (alts < h_km), fracs, 0.0).sum(axis=1)
        return np.where(np.isfinite(fracs).all(axis=1), below, np.nan)


def fit_Cn2_grid(alts, fracs, boundaries_km, layer_altitudes_km=None, pin_ground_layer=True):
    """
    Grid from a set of raw AOF profiles [N, 8]: the nodes are the median altitudes of the layer indices (first node at 0 km), the
    median grid profile is the normalized median of the projected profiles. `boundaries_km` separate the model layers.
    """
    alts  = np.array(alts,  dtype=float, ndmin=2)
    fracs = np.array(fracs, dtype=float, ndmin=2)
    alt_ok = np.isfinite(alts) & (alts >= 0.0) & (alts < MAX_ALTITUDE_KM) & np.isfinite(fracs) & (fracs > 0.0) & (fracs <= 1.0)
    alt_ok[:, 1:] &= alts[:, 1:] > 0.0
    
    layer_alts = np.array([np.median(alts[:, i][alt_ok[:, i]]) for i in range(N_AOF_LAYERS)]) if layer_altitudes_km is None \
                 else np.asarray(layer_altitudes_km, dtype=float)
                 
    nodes = layer_alts.copy()
    nodes[0] = 0.0
    nodes = np.maximum.accumulate(nodes)
    
    if np.any(np.diff(nodes) <= 0):
        raise ValueError(f'Median layer altitudes are not increasing: {nodes}')

    grid = Cn2Grid(nodes, boundaries_km, layer_altitudes_km=layer_alts, pin_ground_layer=pin_ground_layer)
    projected, _, _, _, valid = grid.process(alts, fracs)
    median = np.median(projected[valid], axis=0)
    
    return Cn2Grid(
        nodes,
        boundaries_km,
        median_fracs=median / median.sum(),
        layer_altitudes_km=layer_alts,
        pin_ground_layer=pin_ground_layer
    )
