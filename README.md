# Directional sparse filtering for blind speech separation.

MATLAB Code for the following papers:

K. Watcharasupat, A. H. T. Nguyen, C. -H. Ooi and A. W. H. Khong, "Directional Sparse Filtering Using Weighted Lehmer Mean for Blind Separation of Unbalanced Speech Mixtures," ICASSP 2021 - 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 4485-4489, doi: 10.1109/ICASSP39728.2021.9414336.

A. H. T. Nguyen, V. G. Reju and A. W. H. Khong, "Directional Sparse Filtering for Blind Estimation of Under-Determined Complex-Valued Mixing Matrices," in IEEE Transactions on Signal Processing, vol. 68, pp. 1990-2003, 2020, doi: 10.1109/TSP.2020.2979550.

A. H. T. Nguyen, V. G. Reju, A. W. H. Khong and I. Y. Soon, "Learning complex-valued latent filters with absolute cosine similarity," 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2017, pp. 2412-2416, doi: 10.1109/ICASSP.2017.7952589.

## For Python Version
https://github.com/karnwatcharasupat/directional-sparse-filtering-tf

## Usage
1. Compile the mex files for minFunc by running ./lib/minFunc/mexAll.m
2. Run ./DEMO_TSP2020/DEMO_speech_separation.m or ./DEMO_ICASSP2021/DEMO_speech_separation_lehmer.m to separate some SISEC2011 mixtures
3. (Optionally) Edit the demos separate your own mixtures.
