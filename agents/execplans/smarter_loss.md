# New smarter loss

The goal of this

# Instructions

As you may see in `./datasets/MUSE_STD_dataset/create_STD_dataset.py`, all NFM PSFs are manually labeled by different flags. Here is their meaning:

- "Bad", "medium", or "high quality" refers to how blurry a PSF appers or, in other words, how fomintes it is by HO and TT errors. when two flgas are checked simultaneosuly, it means that the PSF quality is somewhere in-between.

- "Corrupted" spreaks for itself, it basically means that PSF must be discarded from the training set.

- "Non-point source" indictes that the core of the PSF looks distorted or assymetric, which can be caused by the "phase-bump" defect or by quasi-static NCPAs. 

- "Low AO error" means that the PSF region inside the AO halo is quite deep, indicationg a good AO correction. Meanwhile, the regionoutside the corrected radius ca exhiit strong halo.

- "Has streaks" means that at least the TT error is low enougn so it doesn't blur the diffractive streaks of the PSF.

- Finally, the very impirtant on is the "Tomographic error" flag, which indicates the presence of cross-like pattern which appears when higher $C_n^2$ layers have a bigger weight.




The PSFs that has non-point source flag enabled must try to explain assymetry with the phase bump error contributor (the 0th LO NCPAs term)

Low-aulity PSFs are less peak-dominated, so the loss might weight PSF more unifromly in this case

Has streaks implies that TT contribution should be low for this particular sample

The tomographic error should force the loss to emhasize the cross-like feature ouside the PSF core, but before the halo cut-off.


# Goal

Your goal is to design a smarter los function for used for NFM calibrator training. It can take the `BalancedLoss` as a basis, but 