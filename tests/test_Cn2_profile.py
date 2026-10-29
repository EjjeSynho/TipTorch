"""
Checks of data_processing/Cn2_profile.py (CPU, < 1 s, no data needed):
    python tests/test_Cn2_profile.py

1. repair: placeholders / garbage fixed inside the profile, mass renormalized, empty profiles returned as NaN
2. to_grid: mass and (5/3)-moment preserved, pinned ground layer, layers at a node / above the last node
3. process, state_dict round trip, legacy states rejected, fit_Cn2_grid on random profiles
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.Cn2_profile import Cn2Grid, fit_Cn2_grid, N_AOF_LAYERS, MOMENT_POWER


def check(condition, message):
    print(f'[{"PASS" if condition else "FAIL"}] {message}')
    return bool(condition)


def main():
    results = []
    p = MOMENT_POWER
    nodes = np.array([0.0, 8.5, 18.0])
    layer_alts = [0.1, 2.0, 6.5, 11.0, 15.0, 18.5, 22.5, 26.0]
    grid  = Cn2Grid(nodes, layer_alts, pin_ground_layer=False)

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
    results.append(check(g.shape == (4, 3) and np.allclose(g[valid].sum(axis=1), 1.0) and np.isnan(g[2]).all(), 'to_grid: 3 weights, mass preserved, NaN propagated'))
    m_raw  = np.nansum(np.clip(a, 0, nodes[-1])**p * f, axis=1)
    m_grid = (g * nodes**p).sum(axis=1)
    results.append(check(np.allclose(m_raw[valid], m_grid[valid]), 'to_grid: (5/3)-moment preserved (above the top node: clipped)'))
    t2 = (2.0 / 8.5)**p
    g2 = grid.to_grid([[0.0, 2.0] + [8.5]*6], [[0.0, 1.0] + [0.0]*6])
    results.append(check(np.allclose(g2[0], [1 - t2, t2, 0.0]), f'to_grid: a 2 km layer splits {np.round(g2[0], 3).tolist()} onto 0 / 8.5 / 18 km'))
    g_pin = Cn2Grid(nodes, layer_alts, pin_ground_layer=True).to_grid([[0.5, 2.0] + [8.5]*6], [[0.6, 0.4] + [0.0]*6])
    results.append(check(np.allclose(g_pin[0], [0.6 + 0.4 * (1 - t2), 0.4 * t2, 0.0]), 'to_grid: pinned ground layer goes entirely to the 0 km node'))
    g_top = grid.to_grid([[0.0, 8.5, 40.0] + [8.5]*5], [[0.5, 0.2, 0.3] + [0.0]*5])
    results.append(check(np.allclose(g_top[0], [0.5, 0.2, 0.3]), 'to_grid: a layer at a node lands on it, a layer above the last node lands on it'))

    # 3. process, state, fit
    w, n_rep_p, valid_p = grid.process(alts, fracs)
    results.append(check(np.allclose(w[valid], g[valid], equal_nan=True) and valid_p.tolist() == valid.tolist(), 'process: repair + to_grid'))
    grid2 = Cn2Grid.from_state_dict(grid.state_dict())
    results.append(check(np.allclose(grid2.nodes, grid.nodes) and np.allclose(grid2.layer_altitudes, grid.layer_altitudes) and
                         grid2.pin_ground_layer == grid.pin_ground_layer and grid2.feature_names == ['Cn2_grid_1', 'Cn2_grid_2', 'Cn2_grid_3'],
                         'state_dict round trip'))
    try:
        Cn2Grid.from_state_dict({'nodes_km': [0.0, 2.1, 6.6], 'boundaries_km': [4.0, 13.0], 'layer_altitudes_km': layer_alts})
        results.append(check(False, 'legacy state (with boundaries) rejected'))
    except ValueError:
        results.append(check(True, 'legacy state (with boundaries) rejected'))
    rng = np.random.default_rng(0)
    A = np.sort(rng.uniform(0, 28, size=(200, N_AOF_LAYERS)), axis=1); A[:, 0] = rng.uniform(0, 1, 200)
    W = rng.dirichlet(np.ones(N_AOF_LAYERS) * 2, size=200)
    fitted = fit_Cn2_grid(A, W, nodes_km=[0.0, 8.6, 18.3])
    results.append(check(np.allclose(fitted.nodes, [0.0, 8.6, 18.3]) and np.isclose(fitted.median_fracs.sum(), 1.0) and len(fitted.layer_altitudes) == N_AOF_LAYERS,
                         f'fit_Cn2_grid: median profile {np.round(fitted.median_fracs, 3).tolist()}'))

    print(f'\n{sum(results)}/{len(results)} checks passed')
    return 0 if all(results) else 1


if __name__ == '__main__':
    sys.exit(main())
