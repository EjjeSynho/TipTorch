
While studying the residuals of the fitting to the STD stars residuals, I noticed the huge errors on peaks, centered around just a few bright pixels that happen to be exactly at the core.


My hypothesis that this might happen due the overexposure of the brigtest pixels and charge overfill, which cause these pixel to take unphysical values, which are, nevertheless, are not NaN. Hence, such pixels values can vary significantly between spectral slices, which is not connected with the real astrophysical spectral signal. When processing other on-sky NFM observations with the routines from `tools/observations.py`, I actually cover such cases. 

I want you to prove of disprove this hypothesis on what could be the reason behind such bad peak behaviour.

If bad bright pixel is the reason behind poor near-peak fitting accuracy, then it can be explained by 


What speaks for this hypothesis: the rest of the PSF fits super well to the data, apart from just a few pixels in the center
Meanwhile, trying to fit the few central sharp pixels introduces bias the entire core of the PSF model because generated PSFs are "smooth".

Investigate the functions inside `tools/observations.py` like `_detect_bad_exposures()` and connected, see if you can re-use the same logic when processing the STD stars data.



Move code/functions that can be re-used both for observations and STD dataset compiler into the `MUSE_data_utils.py`.

