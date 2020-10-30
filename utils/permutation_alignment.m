% This permutation alignment algorithm implements a simplified version 
% of [2] using linear assignment algorithm, whereby a single centroid 
% is used for each band, cross-band alignments are done in hierarchical
% manner from bottom up. Fine tuning based on harmonic and adjacent
% frequencies [3] is also utilized.
%
% Notes
% -----
% 	This program is distributed only for non-commercial research use
%   at universities, academic institutions, or non-profit organizations. 
%   Commercial uses and redistribution are not permitted without
%   permission from the authors.
%
%   Please kindly cite the following papers if you use or modify any 
%   part of this program:
%
%	  [1] A. H. T. Nguyen, V. G. Reju, and A. W. H. Khong, “Directional Sparse Filtering 
%	  for Blind Estimation of Under-determined Complex-valued Mixing Matrices,” IEEE 
%	  Transactions on Signal Processing, vol. 68, pp. 1990-2003, Mar. 2020.
%
%	  [2] L. Wang, “Multi-band multi-centroid clustering based permutation alignment for 
%   frequency-domain blind speech separation,” Digit. Signal Process., vol. 31, pp. 
%	  79–92, Aug. 2014.
%
%	  [3] H. Sawada, S. Araki, and S. Makino, “Underdetermined convolutive
%	  blind source separation via frequency bin-wise clustering and permutation 
%	  alignment,” IEEE Trans. Audio. Speech. Lang. Processing, vol. 19,
%	  no. 3, pp. 516–527, Jan. 2011.
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
function [P, total_per] = permutation_alignment(P, max_iter, tol)
  if nargin < 1, error('Not enough arguments.'); end
  if nargin < 2, max_iter = 15; end
  if nargin < 3, tol = 0.001; end

  [n_freq, n_frame, n_src] = size(P);
  total_per = repmat((1:n_src), [n_freq 1]);

  % Normalize all sequence to zero-mean unit-variance along time index
  P_normalized = P;
  P_normalized = bsxfun(@minus, P_normalized, mean(P_normalized, 2));

  step = 2;
  % Multi-band single-centroid permutation alignment [2]
  while step < n_freq
    for i = 2:step:n_freq
      group1_idx = i:i+step/2-1;
      if i+step-1 == n_freq-1 && step > 2
        group2_idx = i+step/2:n_freq;
      else
        group2_idx = i+step/2:i+step-1;
      end

      f_1group = reshape(mean(P_normalized(group1_idx,:,:), 1), [n_frame,n_src]);
      f_2group = reshape(mean(P_normalized(group2_idx,:,:), 1), [n_frame,n_src]);
      f_1group = bsxfun(@rdivide, f_1group, std(f_1group, 0, 1)+eps);
      f_2group = bsxfun(@rdivide, f_2group, std(f_2group, 0, 1)+eps);

      Q = 1- f_1group' * f_2group / n_frame;
      per = lapjv(Q); % Solving linear assigment problem => permutation

      P(group1_idx, :, per) = P(group1_idx, :, :);
      P_normalized(group1_idx, :, per) = P_normalized(group1_idx, :, :);
      total_per(group1_idx, per) = total_per(group1_idx, :);
    end
    step = step*2;
  end
  
  % Fine tuning w.r.t adjacent and harmonic frequencies [3]
  iter = 0;
  while 1
    iter = iter + 1;
    old_P_normalized = P_normalized;

    for f = n_freq:-1:2
      % ignore dc
      f_harmonic_ = round([(f-1)/2, 2*(f-1), (f-1)/4, 4*(f-1), (f-1)/8, 8*(f-1), (f-1)/16, 16*(f-1), (f-1)/32, 32*(f-1)]);
      f_harmonic = [f_harmonic_, f_harmonic_+1, f_harmonic_+2];
      f_adjacent = [f-3:f-1, f+1:f+3];

      f_finetuning = unique([f_harmonic, f_adjacent]);
      illegal_freq = f_finetuning < 2 | f_finetuning > n_freq | f_finetuning == f;
      f_finetuning(illegal_freq) = [];

      % Find average Q matrix amongst adjacent and harmonic frequencies
      pf = reshape(old_P_normalized(f, :, :), [n_frame n_src]);
      pf_finetuning = reshape(mean(old_P_normalized(f_finetuning,:,:), 1), [n_frame,n_src]);
      pf_finetuning = bsxfun(@rdivide, pf_finetuning, std(pf_finetuning, 0, 1)+eps);

      Q = 1 - pf'*pf_finetuning/n_frame;
      per = lapjv(Q);

      P(f, :, per) = P(f, :, :);
      P_normalized(f, :, per) = old_P_normalized(f, :, :);
      total_per(f, per) = total_per(f, :);
    end

    % Stop when P wont change anymore.
    relative_error = norm(old_P_normalized(:) - P_normalized(:))/(norm(old_P_normalized(:)+eps));
    stop_flag = (iter > max_iter) || ...
              (iter > 1 && relative_error < tol);
    if stop_flag
      break;
    end
  end

  pf_ac = reshape(mean(P_normalized(2:end,:,:), 1), [n_frame,n_src]);
  pf_ac = bsxfun(@rdivide, pf_ac, std(pf_ac, 0, 1)+eps);
  pf_dc = reshape(P_normalized(1,:,:), [n_frame,n_src]);

  Q = 1 - pf_dc' * pf_ac / n_frame;
  per = lapjv(Q); 

  P(1, :, per) = P(1, :, :);
  P_normalized(1, :, per) = P_normalized(1, :, :);
  total_per(1, per) = total_per(1, :);
end

