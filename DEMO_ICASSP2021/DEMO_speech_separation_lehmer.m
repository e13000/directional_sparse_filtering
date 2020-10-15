clear all; close all;
addpath (genpath('..'));

% Reproducible random stream for parfor
seed = 2017;
rng(seed, 'combRecursive');

o.A_init = [];
o.khl_init = true;
o.max_epoch = 500;
o.opt_tol = 1e-5;
o.prog_tol = 1e-9;
o.corrections = 100;
o.norm_eps = 1e-8;
o.deriv_check = 0;
o.verbose = false;
o.use_mex = true;

ralpha = [0.5, 10];
b_ = 12.5;

NFFT = 2048;
n_overlap = NFFT * 3/4;
window = hamming(NFFT, 'periodic');

n_src = 4;
% n_src = 3;
file_name = ['../audiofiles/dev1_female' int2str(n_src) '_liverec_130ms_1m'];
[data, fs] = audioread([file_name '_mix.wav']);
mixture = data'; % n_chan by n_sample

% Load the source's image at each channel
sources = zeros(n_src, 160000, 2);

for i = 1:n_src
    sources(i, :, :) = reshape(audioread([file_name '_sim_' int2str(i) '.wav']), 1, 160000, 2);
end

tic;
[~, n_sampl] = size(mixture);
X = stft(mixture, NFFT, n_overlap, window);
[n_freq, n_frame, n_chan] = size(X);
P = zeros(n_freq, n_frame, n_src);

% for f = 1:n_freq
parfor f = 1:n_freq
    x_f = squeeze(X(f, :, :)).'; % x_f: n_chan by n_frame

    rng(seed, 'combRecursive'); % For reproducible results
    [~, ~, cos2, Jm] = dsf_lite_lehmer(x_f, n_src, ralpha, o);
    P(f, :, :) = calc_softmask(cos2, b_)';
end

[P, ~] = permutation_alignment(P, 10, 1e-3);
P = repmat(P, [1, 1, 1, n_chan]);
Y = permute(repmat(X, [1, 1, 1, n_src]), [1 2 4 3]) .* P;
y_t_anh = istft(Y, n_sampl, NFFT, n_overlap, window);
elapse_time = toc;

[SDR_anh, ISR_anh, SIR_anh, SAR_anh, perm_anh] = bss_eval_images(y_t_anh, sources);

my_result = mean(SDR_anh);
my_result_SIR = mean(SIR_anh);

fprintf(['\nSolving BSS for ' file_name 'in ' int2str(elapse_time) 's']);
fprintf('\nSDR|SIR|ISR|SAR:  %.2f | %.2f | %.2f| %.2f\n', mean(SDR_anh), mean(SIR_anh), mean(ISR_anh), mean(SAR_anh));

for i = 1:n_src
    out = squeeze(y_t_anh(i, :, :));
    audiowrite([file_name '_out_' int2str(i) '.wav'], 0.9 * out / max(abs(out(:))), fs);
end

