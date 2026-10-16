"""
Checks of the k_c-mismatch handling of the MUSE NFM STD stars (samples recorded with fewer corrected DM modes, flagged 'k_c mismatch'
in muse_df.pickle by datasets/MUSE_STD_dataset/detect_kc_mismatch.py):

  1. group_kc_fractions groups the measured k_c fractions by the closest averaged value within the tolerance;
  2. LoadSTDStarCache scales the DM pitch of a flagged sample's config (k_c fraction round trip), nominal samples are untouched;
  3. MultipleTargetsInDifferentObservations refuses to merge configs with different k_c (the model holds one k_c per batch);
  4. KcGroupBatchSampler never mixes k_c groups in a batch and covers every position exactly once;
  5. a TipTorch model updated to a scaled-pitch config with Update(grids=True) reproduces a model built from that config
     (this is what NFMCalibratorTrainer.run_model does when the k_c of the batch changes).

Requires the STD-star cache (muse_df.pickle with the 'k_c fraction' / 'k_c group' columns, cached_cubes/) and a GPU (~1 min).
Run:  python tests/test_kc_mismatch_groups.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'datasets' / 'MUSE_STD_dataset'))

import pickle
import numpy as np
import torch

from STD_dataset_utils import (STD_FOLDER, LoadSTDStarCache, group_kc_fractions, apply_kc_fraction, kc_fraction_from_config,
                               KC_FRACTION_FEATURE, KC_GROUP_FEATURE, KC_GROUP_TOLERANCE)
from machine_learning.calibrators.NFM_calibrator import KcGroupBatchSampler
from tiptorch.managers.config_manager import MultipleTargetsInDifferentObservations
from tiptorch.PSF_models.NFM_wrapper import PSFModelNFM

device = torch.device(sys.argv[1]) if len(sys.argv) > 1 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
results = []

def check(name, ok, info=''):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f'  ({info})' if info else ''))


# 1. Grouping ------------------------------------------------------------------------------------------------------------------
rng = np.random.default_rng(0)
x = np.concatenate([0.905 + 0.008 * rng.standard_normal(14), 0.935 + 0.007 * rng.standard_normal(16), 0.972 + 0.008 * rng.standard_normal(8)])
groups, means = group_kc_fractions(x)
within = np.all(np.abs(x - means[groups - 1]) <= KC_GROUP_TOLERANCE)
check('group_kc_fractions: every sample within the tolerance of its group mean', within, f'{len(means)} groups, means {np.round(means, 3).tolist()}')
check('group_kc_fractions: group ids sorted by increasing k_c', np.all(np.diff(means) > 0) and groups.min() == 1 and groups.max() == len(means))
g1, m1 = group_kc_fractions(np.array([0.93]))
check('group_kc_fractions: single sample -> one group', len(m1) == 1 and g1.tolist() == [1])

# 2. Config scaling -------------------------------------------------------------------------------------------------------------
with open(STD_FOLDER / 'muse_df.pickle', 'rb') as f:
    muse_df = pickle.load(f)
assert KC_FRACTION_FEATURE in muse_df.columns, 'muse_df.pickle has no k_c columns, run datasets/MUSE_STD_dataset/detect_kc_mismatch.py first'

flagged = muse_df.index[muse_df[KC_GROUP_FEATURE] > 0]
id_mis  = int(flagged[np.argmin(muse_df.loc[flagged, KC_FRACTION_FEATURE].values)]) # the strongest mismatch
id_nom  = int(muse_df.index[(muse_df[KC_GROUP_FEATURE] == 0) & ~muse_df['Corrupted'] & ~muse_df['Bad quality']][50])
wvl_ids = [0, 15, 30]

PSF_mis, _, _, cfg_mis = LoadSTDStarCache([id_mis], derotate_PSF=True, normalize=True, subtract_background=True, wvl_ids=wvl_ids, ensure_odd_pixels=True, device=device)
PSF_nom, _, _, cfg_nom = LoadSTDStarCache([id_nom], derotate_PSF=True, normalize=True, subtract_background=True, wvl_ids=wvl_ids, ensure_odd_pixels=True, device=device)
frac_expected = float(muse_df.loc[id_mis, KC_FRACTION_FEATURE])
check(f'LoadSTDStarCache: flagged sample {id_mis} gets k_c = {100 * frac_expected:.1f} % via the DM pitch',
      np.isclose(kc_fraction_from_config(cfg_mis[0]), frac_expected, atol=1e-6), f"DmPitchs = {cfg_mis[0]['DM']['DmPitchs']}")
check(f'LoadSTDStarCache: nominal sample {id_nom} keeps the nominal pitch', np.isclose(kc_fraction_from_config(cfg_nom[0]), 1.0, atol=1e-6))
cfg_round = apply_kc_fraction(apply_kc_fraction(pickle.loads(pickle.dumps(cfg_nom[0])), 0.9), 1.0 / 0.9)
check('apply_kc_fraction round trip', np.isclose(kc_fraction_from_config(cfg_round), 1.0, atol=1e-9))

# 3. Merge guard ----------------------------------------------------------------------------------------------------------------
try:
    MultipleTargetsInDifferentObservations([cfg_mis[0], cfg_nom[0]], device=device)
    check('MultipleTargetsInDifferentObservations refuses mixed k_c', False)
except ValueError as e:
    check('MultipleTargetsInDifferentObservations refuses mixed k_c', 'k_c' in str(e))
try:
    _ = MultipleTargetsInDifferentObservations([cfg_mis[0], pickle.loads(pickle.dumps(cfg_mis[0]))], device=device)
    check('MultipleTargetsInDifferentObservations merges equal k_c', True)
except ValueError as e:
    check('MultipleTargetsInDifferentObservations merges equal k_c', False, str(e))

# 4. Batch sampler --------------------------------------------------------------------------------------------------------------
groups_s = np.array([0] * 40 + [1] * 5 + [2] * 3 + [0] * 7)
for shuffle in (False, True):
    torch.manual_seed(1)
    sampler = KcGroupBatchSampler(groups_s, batch_size=16, shuffle=shuffle)
    batches = list(iter(sampler))
    pure    = all(len(set(groups_s[b].tolist())) == 1 for b in batches)
    cover   = sorted(sum(batches, [])) == list(range(len(groups_s)))
    sizes   = sorted(len(b) for b in batches)
    check(f'KcGroupBatchSampler(shuffle={shuffle}): pure batches, full coverage, {len(sampler)} batches',
          pure and cover and len(batches) == len(sampler) == 5 and sizes == [3, 5, 15, 16, 16], f'sizes {sizes}')

# 5. Update(grids=True) to a new k_c == fresh model ------------------------------------------------------------------------------
model_kwargs = dict(multiple_obs=True, LO_NCPAs=True, chrom_defocus=False, use_Moffat=False, Z_mode_max=9, N_spline_nodes=5, device=device)
cfg_mis_nominal = apply_kc_fraction(pickle.loads(pickle.dumps(cfg_mis[0])), 1.0 / frac_expected) # the same sample with the nominal pitch

with torch.no_grad():
    model_fresh = PSFModelNFM([cfg_mis[0]], **model_kwargs)
    PSF_fresh   = model_fresh(model_fresh.inputs_manager.to_dict()).detach().clone()
    kc_fresh, nOtf_AO_fresh = float(model_fresh.model.kc), int(model_fresh.model.nOtf_AO)

    model_upd = PSFModelNFM([cfg_mis_nominal], **model_kwargs)
    PSF_before = model_upd(model_upd.inputs_manager.to_dict()).detach().clone()
    kc_before, nOtf_AO_before = float(model_upd.model.kc), int(model_upd.model.nOtf_AO)

    model_upd.model.config = MultipleTargetsInDifferentObservations([cfg_mis[0]], device=device) # as run_model does per batch
    model_upd.model.Update(grids=True, pupils=False, tomography=True)
    PSF_after = model_upd(model_upd.inputs_manager.to_dict()).detach().clone()

check('k_c of the updated model follows the scaled pitch', np.isclose(float(model_upd.model.kc), kc_fresh) and np.isclose(kc_fresh / kc_before, frac_expected, atol=1e-6),
      f'k_c {kc_before:.3f} -> {float(model_upd.model.kc):.3f} 1/m, nOtf_AO {nOtf_AO_before} -> {int(model_upd.model.nOtf_AO)} (fresh {nOtf_AO_fresh})')
rel = ((PSF_after - PSF_fresh).abs().amax() / PSF_fresh.abs().amax()).item()
check('Update(grids=True) reproduces the model built from the scaled config', rel < 1e-5, f'max |diff| / peak = {rel:.2e}')
chg = ((PSF_after - PSF_before).abs().amax() / PSF_before.abs().amax()).item()
check('the k_c change modifies the PSF', chg > 1e-3, f'max |diff| / peak = {chg:.2e}')

# the ring diagnostic of detect_kc_mismatch applied to (scaled model as the "data") - (nominal model as the "fit"), the situation of the
# flagged on-sky samples: the ring must sit at frac_expected x the nominal correction radius
from detect_kc_mismatch import ring_profiles, ring_score
prof, coh, _ = ring_profiles(PSF_after[0].cpu().numpy(), PSF_before[0].cpu().numpy(), model_upd.λ_sim.cpu().numpy())
sc = ring_score(prof, coh)
check('nominal-vs-scaled model residual shows the ring at the scaled k_c', abs(sc['kc_fraction'] - frac_expected) < 0.03 and sc['coherence'] > 0.9,
      f"ring at {100 * sc['kc_fraction']:.1f} % (expected {100 * frac_expected:.1f} %), amp {sc['ring_amp']:.2f}")

print(f"\n{sum(results)} / {len(results)} checks passed")
sys.exit(0 if all(results) else 1)
