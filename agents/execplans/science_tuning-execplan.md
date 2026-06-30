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