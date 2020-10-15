% function [What, Ahat, cos2, J] = dsf_lite(X, N, r, o)
%
% Estimate the mixing matrix from mixtures of
% zero-mean highly super Gaussian signals
%
% Parameters
% ----------
%     X: M by K of type double or single
%         Each column is a sample.
%     N: int
%         Number of latent factors/sources
%     r: double
%         Hyper parameter, r<1.
%         r =-Inf is much faster and uses less memory. However, the gradient
%	        is not exact in this case because the cost is non-smooth.
%     options: struct
%       - A_init: (default=[])
%           Initial mixing matrix
%       - khl_init: bool (default=true)
%           Run K-hyperlines to initialize the algorithm
%       - max_epoch: int (default=500)
%           Number of iteration.
%       - opt_tol: double (default=1e-5)
%           Stopping condition, minimum L_inf distance of current parameters
%           and previous parameters.
%       - prog_tol: double (default=1e-9)
%           Stopping condition, minimum L_inf distance of current cost
%           and previous cost.
%       - norm_eps: double (default=1e-8)
%           A small constant to prevent div-by-zero
%       - deriv_check: int (default=0)
%           Fast derivative check if deriv_check == 1, slow derivative check if
%           deriv_check == 2. Otherwise, no check.
%       - verbose: bool (default=true)
%          Set to false to suppress the printing
%       - use_mex: bool (default=true)
%          Use mex file for minFunc
% Returns
% -------
%     What: M by N
%       Demixing matrix (determined cases)
%     Ahat: M by N
%       Estimated mixing matrix
%     cos2: N by K
%       Magnitude square cosine similarity between the Kth sample
%       and all the columns of A
%     J: epoch by 1
%       Cost at each epoch
%
% Notes
% -----
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
function [What, Ahat, cos2, J] = dsf_lite(X, N, r, o)
    if nargin < 4, o = struct; end
    if nargin < 3, r = -1; end
    if r >= 1, error('r must be smaller than 1!'); end

    defaults = {'A_init', [], ...
                'khl_init', true, ...
                'max_epoch', 500, ...
                'opt_tol', 1e-5, ...
                'prog_tol', 1e-9, ...
                'corrections', 100, ...% LBFGS
                'norm_eps', 1e-8, ...
                'deriv_check', 0, ...
                'verbose', false, ...
                'use_mex', true, ...
                };
    o = parse_options(o, defaults);
    [M, K] = size(X);

    %-- Whitening
    [X, Q_PCA, Q_IPCA] = whitening(X);

    %-- Normalize X
    column_norm_X = sqrt(sum(conj(X) .* X, 1) + o.norm_eps);
    X = bsxfun(@rdivide, X, column_norm_X);

    %-- Initialization
    if isempty(o.A_init)
        A_init = randn(M, N);

        if ~isreal(X)
            A_init = A_init + 1i * randn(M, N);
        end

    else
        A_init = o.A_init;
    end

    J_khl = [];

    if o.khl_init
        % Run K-hyperlines to initialize the algorithm
        o.A_init = A_init;
        [~, A_init, ~, J_khl, ~] = k_hyperlines(X, N, o);
    end

    A_init = cast(A_init, class(X));
    column_norm_A_init = sqrt(sum(conj(A_init) .* A_init, 1) + o.norm_eps);
    A_init = bsxfun(@rdivide, A_init, column_norm_A_init);

    % Minimize the objective function of DSF
    obj_func = @(A, X) cost_function(A, X, r, o.norm_eps);
    [best_A, J] = optimize(obj_func, A_init, X, o.max_epoch, o.opt_tol, ...
        o.prog_tol, o.corrections, o.deriv_check, ...
        o.use_mex, o.verbose);
    % Normalization
    [U, S, V] = svd(best_A, 'econ');
    best_A = U * V'; % best_A = (best_A*best_A')^-1/2*best_A;
    column_norm_best_A = sqrt(sum(conj(best_A) .* best_A, 1) + o.norm_eps);
    best_A = bsxfun(@rdivide, best_A, column_norm_best_A);

    % Returns
    % J = [J_khl; J];
    What = best_A' * Q_PCA; % Valid for determined case only
    Ahat = Q_IPCA * best_A;
    cos2 = abs(best_A' * X).^2;
end

function [cost, grad] = cost_function(A, X, r, norm_eps)
    [M, N] = size(A);
    [~, K] = size(X);

    Aorth = A;
    [U, S, V] = svd(A, 'econ');
    % row-wise decoupling
    Aorth = U * V'; % Aorth = (A*A')^-1/2*A;

    % column norm normalization
    column_norm_A = sqrt(sum(conj(Aorth) .* Aorth, 1) + norm_eps);
    Anorm = bsxfun(@rdivide, Aorth, column_norm_A);

    % Compute loss and gradient
    angle_ = Anorm' * X;
    Y2 = 1 - abs(angle_).^2;
    [Y2_min, min_idx] = min(Y2, [], 1);

    if r == -Inf
        g = sum(Y2_min);
        dg = -sparse(min_idx, 1:K, 1, N, K);
    elseif r == 0
        Y2_prod = prod(Y2 + norm_eps).^(1 / N);
        g = sum(Y2_prod);
        dg =- bsxfun(@rdivide, Y2_prod, Y2 + norm_eps) / N;
    else
        Y2_power_mean = Y2_min .* ...
            sum((bsxfun(@rdivide, Y2, Y2_min)).^r, 1).^(1 / r) / N^(1 / r);
        % Handling the case when the smallest value at a column is 0
        Y2_power_mean(Y2_min == 0) = 0;
        g = sum(Y2_power_mean);

        dg = -bsxfun(@rdivide, Y2_power_mean, Y2).^(1 - r) / N;
        dg(:, Y2_min == 0) = 0;
        dg(min_idx(Y2_min == 0), Y2_min == 0) =- 1 / N^(1 / r);
    end

    cost = (1 / K) * g;
    grad = 2 * (1 / K) * X * (angle_ .* dg)';
    grad = cast(grad, class(A));

    % Apply the chain rule through column norm normalization
    grad = grad - bsxfun(@times, Anorm, sum(real(conj(Anorm) .* grad), 1));
    grad = bsxfun(@rdivide, grad, column_norm_A);

    % Apply the chain rule through row-wise decoupling
    Sinv = diag(1 ./ diag(S));
    C = -(Sinv * U' * grad * V) ./ (bsxfun(@plus, diag(S), diag(S)'));
    grad = U * (C' + C) * S * V' + U * Sinv * U' * grad;
end
