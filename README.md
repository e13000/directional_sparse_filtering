# Directional sparse filtering for blind speech separation.

MATLAB Code of the following paper:

K. Watcharasupat, A. H. T. Nguyen, C.-H. Ooi, and A. W. H. Khong, "Directional
Sparse Filtering using Weighted Lehmer Mean for Blind Separation of Unbalanced
Speech Mixtures," submitted to ICASSP2021.

A. H. T. Nguyen, V. G. Reju, and A. W. H. Khong, “Directional Sparse Filtering 
for Blind Estimation of Under-determined Complex-valued Mixing Matrices,” IEEE 
Transactions on Signal Processing, vol. 68, pp. 1990-2003, Mar. 2020.

## Usage
1. Compile the mex files for minFunc by running ./lib/minFunc/mexAll.m
2. Run ./DEMO_TSP2020/DEMO_speech_separation.m or ./DEMO_ICASSP2021/DEMO_speech_separation_lehmer.m to separate some SISEC2011 mixtures
3. (Optionally) Edit the demos separate your own mixtures.
