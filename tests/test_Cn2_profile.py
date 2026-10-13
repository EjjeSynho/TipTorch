"""
Checks of data_processing/Cn2_profile.py (CPU, < 1 s, no data needed):
    python tests/test_Cn2_profile.py

1. repair: placeholders / garbage fixed inside the profile, mass renormalized, empty profiles returned as NaN
2. to_grid: mass and mean altitude preserved, layers above the last node land on it
3. to_layers: bin masses, (5/3)-moment heights inside the bins, ground layer at 0, empty bins get the median heights
4. state_dict round trip, fit_Cn2_grid on random profiles
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.Cn2_profile import Cn2Grid, fit_Cn2_grid, N_AOF_LAYERS


def check(condition, message):
    print(f'[{"PASS" if condition else "FAIL"}] {message}')
    return bool(condition)


def main():
    results = []
    nodes = [0.0, 2.0, 6.5, 11.0, 15.0, 18.5, 22.5, 26.0]
    grid  = Cn2Grid(nodes, boundaries_km=[4.0, 13.0], layer_altitudes_km=[0.1, 2.0, 6.5, 11.0, 15.0, 18.5, 22.5, 26.0], pin_ground_layer=False)

    # 1. repair
    alts  = np.array([[0.1, 2.0, 7.0, 12.0, 16.0, 1000.0, 0.0, 0.0],   # placeholders in the empty upper layers
                      [0.5, 3.0, 9.0, 12.5, 505.0, 7.4, 8.7, 12.3],    # one layer with mass at a garbage altitude, unsorted layers
                      [np.nan]*8,                                      # missing profile
                      [0.1, 2.0, 7.0, 12.0, 16.0, 19.0, 23.0, 27.0]])
    fracs = np.array([[0.5, 0.3, 0.1, 0.06, 0.04, 0.0, 0.0, 0.0],
                      [0.7, 0.17, 0.03, 0.03, 0.02, 0.014, 0.02, 1e21],
                      [np.nan]*8,
                      [0.3, 0.3, 0.1, 0.1, 0.1, 0.05, 0.03, 0.02]])
    a, f, n_rep, valid = grid.repair(alts, fracs)
    results.append(check(valid.tolist() == [True, True, False, True], f'repair: validity {valid.tolist()}'))
    results.append(check(np.allclose(np.nansum(f, axis=1)[valid], 1.0), 'repair: fractions renormalized to 1'))
    results.append(check(n_rep.tolist() == [0, 1, 0, 0], f'repair: repaired layers with mass {n_rep.tolist()}'))
    results.append(check(a[1, 4] == 15.0 and f[1, 7] == 0.0, 'repair: garbage altitude -> median altitude of the layer, garbage fraction -> empty layer'))
    results.append(check(np.isnan(a[2]).all() and np.isnan(f[2]).all(), 'repair: missing profile stays NaN'))

    # 2. grid
    g = grid.to_grid(a, f)
    results.append(check(np.allclose(g[valid].sum(axis=1), 1.0) and np.isnan(g[2]).all(), 'to_grid: mass preserved, NaN propagated'))
    m_raw  = np.nansum(np.clip(a, 0, nodes[-1]) * f, axis=1)
    m_grid = (g * np.asarray(nodes)).sum(axis=1)
    results.append(check(np.allclose(m_raw[valid], m_grid[valid]), 'to_grid: mean altitude preserved (above the top node: clipped)'))
    g_pin = Cn2Grid(nodes, [4.0, 13.0], layer_altitudes_km=grid.layer_altitudes, pin_ground_layer=True).to_grid(a, f)
    results.append(check(np.isclose(g_pin[0, 0], f[0, 0]) and np.isclose(g_pin[0, 1], 0.3), 'to_grid: pinned ground layer goes entirely to the 0 km node'))
    g_top = grid.to_grid([[0.0, 2.0, 6.5, 11.0, 15.0, 18.5, 22.5, 40.0]], [[0.1]*7 + [0.3]])
    results.append(check(np.isclose(g_top[0, -1], 0.3) and np.isclose(g_top[0, -2], 0.1), 'to_grid: a layer above the last node lands on it'))

    # 3. layers
    w, h = grid.to_layers(g)
    results.append(check(w.shape == (4, 3) and np.allclose(w.sum(axis=1), 1.0) and np.all(h[:, 0] == 0.0), 'to_layers: 3 normalized weights, GL at 0 km'))
    results.append(check(np.all(h[valid, 1] >= 4.0) and np.all(h[valid, 1] <= 13.0) and np.all(h[valid, 2] >= 13.0), 'to_layers: heights inside the bins'))
    # profile 0 by hand: 0.1 km -> 0.475 / 0.025 on the nodes 0 / 2 km, 2 km -> 0.3, 7 km -> 0.0889 / 0.0111, 12 km -> 0.045 / 0.015, 16 km -> 0.0286 / 0.0114
    results.append(check(np.allclose(w[0], [0.8, 0.145, 0.055], atol=1e-3), f'to_layers: weights are the bin masses {np.round(w[0], 4).tolist()}'))
    w_e, h_e = grid.to_layers([[1.0] + [0.0]*7])
    results.append(check(np.allclose(w_e[0], [1, 0, 0]) and np.allclose(h_e[0, 1:], grid._median_heights[1:]), 'to_layers: empty bins get the median heights'))
    w_nan, h_nan = grid.to_layers([[np.nan]*8])
    w_def, h_def = grid.default_layers()
    results.append(check(np.allclose(w_nan[0], w_def) and np.allclose(h_nan[0], h_def), 'to_layers: NaN profile -> median layers'))

    # 4. state round trip and fit
    grid2 = Cn2Grid.from_state_dict(grid.state_dict())
    results.append(check(np.allclose(grid2.nodes, grid.nodes) and np.allclose(grid2.membership, grid.membership), 'state_dict round trip'))
    rng = np.random.default_rng(0)
    A = np.sort(rng.uniform(0, 28, size=(200, N_AOF_LAYERS)), axis=1); A[:, 0] = rng.uniform(0, 1, 200)
    W = rng.dirichlet(np.ones(N_AOF_LAYERS) * 2, size=200)
    fitted = fit_Cn2_grid(A, W, boundaries_km=[4.0, 13.0])
    results.append(check(fitted.nodes[0] == 0.0 and np.all(np.diff(fitted.nodes) > 0) and np.isclose(fitted.median_fracs.sum(), 1.0),
                         f'fit_Cn2_grid: nodes {np.round(fitted.nodes, 1).tolist()}'))

    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
