%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
function [X, Q_PCA, Q_IPCA] = whitening(X)
  [M, K] = size(X);

  X = bsxfun(@minus, X, mean(X, 2));
  covX = X*X' / (K-1);

  [E, D, ~] = svd(covX);

  D_isqrtm = diag(real(1./sqrt(max(diag(D), 1e-16))));
  D_sqrtm = diag(real(sqrt(max(diag(D), 1e-16))));

  Q_PCA = E*D_isqrtm*E'; % ZCA
  Q_IPCA = E*D_sqrtm*E';
  X = Q_PCA * X;
end
