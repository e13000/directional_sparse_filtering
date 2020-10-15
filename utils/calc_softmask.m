% 	This program is distributed only for non-commercial research use
%   at universities, academic institutions, or non-profit organizations. 
%   Commercial uses and redistribution are not permitted without
%   permission from the authors.
%
%   Please kindly cite the following paper if you use or modify any 
%   part of this program:
%
%	  A. H. T. Nguyen, V. G. Reju, and A. W. H. Khong, “Directional Sparse Filtering 
%	  for Blind Estimation of Under-determined Complex-valued Mixing Matrices,” IEEE 
%	  Transactions on Signal Processing, vol. 68, pp. 1990-2003, Mar. 2020.
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.

function [mask] = calc_softmask(cos2, beta_)
  [N, K] = size(cos2);

  [cos2_max, cos2_max_idx] = max(cos2, [], 1);

  if beta_ == Inf
    mask = sparse(cos2_max_idx, 1:K, 1, N, K);
  else
    mask = exp(beta_ * bsxfun(@minus, cos2, cos2_max));
    mask = bsxfun(@rdivide, mask, sum(mask, 1));
  end
end

