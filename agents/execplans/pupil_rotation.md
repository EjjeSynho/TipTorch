# New pupil rotation mechanism for TipTorch

I would like to ask you to do a massive change in TipTorch to enable for suport of multiple rotation angles inside a single model instance.


## Context

- As you may notices, the `NFMWrapper` support for two regimes: multiple observations stacked together and multiple objects from a single observation stacked together.
In the first case, all objects must share the same pupil rotation angle. This limitation comes from TipTorch, where all trms share the same rotation matrix.

- In MUSE NFM, pupil rotation coincides with the DM rotation as DM is pre-focal. As a result, for different pupil/DM rotation angles, the focal-plane PSF also rotates.

- Worth noting that on other instruments, it might not be true if DM is post-focal, then pupil rotation and DM rotation angles differ. Anyway, think about it and cross-check my conclusion about it.

- When calibrating the model on dfferent STD stars coming from different OBs and evelations, this is leveraged by derotating the on-sky PSFs, which in turn can damage the peak and intriduce unexplained variance into the calibrattor training.


## Goal

Your goal is to add a per-OB rotation angle support into TipTorch model. It includes the following steps:

- Before doing anything related to angles, check consistency between the axes direction for `dx/y`, `src_dirs_x/y`, `U/V`, and `k_x/y`. The visuals tests done in `development/PSF_explorer.py` reveled that PSF shape elongates in the perpendicular direction to the one in the field position when going off-axis, which should not be the case. The error is inside TipTorch or inside the PSF explorer. Anyway, there is some inconsistency of the coordinates going on, so it's worth checking.

- Once this is clear and fixed, make it possible to input a tensor of pupil angles.

- Initalize a per-OB rotation matrix. I'm referring to OBs now since we're mosty adreessing the multiple objects == multiple OBs scenario here. For multiple objects sharing the smae OB, the pupil values must remain singular.

- Initialize a sequence of rotation matrices per-OB.

- To save memory, and voising pre-computing frequency grids for every object, now it be adressed not like `self.k`, but as `self.k_(angle)` just like it's implemented with `self.r0_(airmass)` currently. When multiple angles provided, this function will return a cube of frequency grids with `N_obs` dimension, with every grid rotated accordingly. If single angle is provided, it will just return singular grids.

- Do the same for other grids like `U, V` if necessary (make the rotatuon consistent with the phase generator, then).

- Do not introduce DM angle as a separate entity just yet. It will come later.

- Make sure to change the handling of the config initialization in this case, since currently pupil angle is handles as a singular value. Investigate and clear this.

- Also look into the data preparation routines and calibrator training/validation for MUSE NFM to account for the new non-derotated scenario.

- What is **important**, is that if only a single pupil angle values is provided, everything in the code must behave as it's behaving now without being slower.


## Practical instructions

- As you may noticed, there are several ways to deliver pupil/phase masks to TipTorch. If loaded from FITS, TipTorch must create a rotated copy of pupil and apodizer masks or every angle values. If a pupil/static phase generator is passed as input, the generator must handle the different angles while TipTorch must provide it angle values.

Perhaps, you may find a better approach to implment the requested functionality. Anyway, try not to breed to much code and ocusof conciseness and performance.