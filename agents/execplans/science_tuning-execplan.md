## Problem statement
The goal of this plan is to allow fine-tuning the NFM calibrator NN directly on the science data. This is usefull since the STD calibration dataset usually uses bright on-axis stars, while the science data can contain some faint stars captured off-axis. Therefore, it is absolutely essential to utilize the functionality of `MUSEObservation` class. The goal is that users can run fine-tuning on their data and then re-use fune-tuned calibrators for other similar science cases.


## Implementation plan

1. First, read the `machine_learning/train_NFM_calibrator.py`. In that routine, the training happens on the isolated bright STD stars with pre-normalized PSFs.

2. Check in `machine_learning/calibrators/NFM_calibrator` if you can utilize existing funtionality of the `NFMCalibrator` class. Perhaps, the fine-tuning loop can be a part of this class somehow?

3. Look at the `tools/observations/MUSEObservation`. Fine-tuning must be similar to the PSF model optimization implemented in this class in ints "phylosophy". However, instead of the PSF model inputs, thw calibrator NN's weights must be optimized. Meawhile, call to draw PSFs must be done in the OB fashion, not in the STD trainer fashion, since real objects can be overlapping, have different brigtness within the same field, placed off-axis, and so on, compared to the idealized pre-normalized STD calibration targets.

4. Then, take a look at the `development/MUSE_OB.py`, `development/MUSE_quasar.py` and `development/MUSE_omega_cen.py`. These files contain examples of how to use `MUSEObservation` class.

5. Implement the fine-tuning loop inside `MUSEObservation` and maybe partially in `NFMCalibrator`.

6. Finally, use the fine-tuning function inside the `development/MUSE_quasar.py`.


## Pay attention to the following

- Since tuning is done on a very few objects, please ensure that overfitting does not occure, design and regularize the fine-tuning loop accordingly
- Note, that the calibrator NN is tiny, so it might be prone to forgetting
- Tuning must be done with Adam or similar methods in the fashion like in STD trainer, not with the 2nd-order L-BFGS
- When tuning the calibrator, make sure that the tuned version is stored separately and does not overwrite the original calibrator weights

Append your thoughts and conclusions to the end of this fle. Good luck!

## Implementation notes and conclusions

- Implemented science-data fine-tuning as `MUSEObservation.FineTuneCalibrator(...)`, rather than reusing the STD training loader. The loop uses the observation's normal sparse-field rendering path: calibrator telemetry prediction -> PSF model inputs -> per-source PSFs -> shared OB canvas with overlapping sources.
- The optimizer is `AdamW` with low default learning rate, weight decay, gradient clipping, ReduceLROnPlateau scheduling, and early stopping. This matches the requested first-order training style and avoids the L-BFGS optimizer used for direct PSF-model fitting.
- To reduce overfitting and forgetting, only selected calibrator output dimensions are allowed to affect the rendered model; untuned outputs are clamped to the frozen reference prediction. The network is also regularized toward its starting weights.
- The regularization is now intentionally adjustable: `regularization_looseness > 1` divides both anchoring penalties, while `prediction_anchor` and `weight_anchor` can still be set directly. The defaults were relaxed from the initial conservative values so science fine-tuning can visibly move the NN prediction.
- `NFMCalibrator` now keeps the loaded bundle state and can save a new bundle via `save(...)`. Science-tuned weights are written to a separate path by default, so the original STD-trained calibrator is not overwritten.
- `development/MUSE_quasar.py` now calls `ob.FineTuneCalibrator(...)` after `InitSimulation()` and before `FitPSFModel(...)`, saving a science-tuned bundle for reuse on similar observations.
- Verification performed here: Python syntax compilation for the touched modules. Full runtime validation still needs the local MUSE science files plus the baseline calibrator bundle.
