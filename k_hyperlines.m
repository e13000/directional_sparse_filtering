% Ref:
%   - P. D. O’Grady and B. A. Pearlmutter, Hard-LOST: Modified k-means for
%   oriented lines," in Proc. Irish Signals Syst. Conf., 2004, pp. 247-252.
%   - Z. He, A. Cichocki, Y. Li, S. Xie, and S. Sanei, K-hyperline clustering
%   learning for sparse component analysis," Signal Processing, vol. 89, no. 6,
%   pp. 1011-1022, Jun. 2009.

%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
function [What, Ahat, cos2, J, C] = k_hyperlines(X, N, o)
  if nargin < 3, o = struct; end
  defaults = {'verbose', false, 'norm_eps', 1e-8, ...
              'A_init', [], 'max_epoch', 500, 'prog_tol', 1e-9};
  o = parse_options(o, defaults);
  [M, K] = size(X);

  [X, Q_PCA, Q_IPCA] = whitening(X);

  % Normalization
  column_norm_X = sqrt(sum(conj(X) .* X, 1) + o.norm_eps);
  X = bsxfun(@rdivide, X, column_norm_X);

  % Initialization
  if isempty(o.A_init)
      A = randn(M, N);
    if ~isreal(X)
      A = A+1i*randn(M, N);
    end
  else
    A = o.A_init;
  end
  A = cast(A, class(X));
  column_norm_A = sqrt(sum(conj(A) .* A, 1)+o.norm_eps);
  A = bsxfun(@rdivide, A, column_norm_A);

  J = zeros(o.max_epoch, 1);
  for epoch = 1:o.max_epoch
    % Compute the distance
    D = 1 - abs(A'*X).^2;
    % Membership
    [Dmin, C_min_idx] = min(D, [], 1);
    C = sparse(C_min_idx, 1:K, 1, N, K);

    % Compute the cost
    scale = 1;
    J(epoch) = sum(Dmin) * scale / K; % cost = sum(sum(C .* D));
    if o.verbose
      fprintf('iter: %d,\t\tcost: %e\n', epoch, J(epoch));
    end
    if ((epoch > 1) && (abs(J(epoch-1) - J(epoch)) < o.prog_tol))
      J = J(1:epoch);
      break;
    end

    % Find the centroids
    old_A = A;
    for j = 1:N
      vv = X(:,C_min_idx==j);
      R_j = vv*vv'; % To make sure R_j is self-adjoint

      % Eig is faster than svd since R_j is symmetric; 
      % however, svd is more numerically stable
      [e_R_j,~,~] = svd(R_j);
      A(:,j) = e_R_j(:,1);
    end
  end

  What = A'*Q_PCA;
  Ahat = Q_IPCA*A; 
  cos2 = abs(A'*X).^2;
end
