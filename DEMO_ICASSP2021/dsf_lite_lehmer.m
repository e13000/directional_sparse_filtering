% function [What, Ahat, cos2, J] = dsf_lite_lehmer(X, N, ralpha, o)
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
%     ralpha: [double, double]
%         Hyperparameters r and alpha, 0 < r < 1, alpha >= 0.
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
%     K. Watcharasupat, A. H. T. Nguyen, C.-H. Ooi, and A. W. H. Khong, "Directional
%     Sparse Filtering using Weighted Lehmer Mean for Blind Separation of Unbalanced
%     Speech Mixtures," submitted to ICASSP2021.
%
%     A. H. T. Nguyen, V. G. Reju, and A. W. H. Khong, “Directional Sparse Filtering
%     for Blind Estimation of Under-determined Complex-valued Mixing Matrices,” IEEE
%     Transactions on Signal Processing, vol. 68, pp. 1990-2003, Mar. 2020.
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg) and Karn Watcharasupat (karn001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen and Karn Watcharasupat. All rights reserved.
function [What, Ahat, cos2, J] = dsf_lite_lehmer(X, N, ralpha, o)
    if nargin < 4, o = struct; end
    if nargin < 3, ralpha = [0.5, 10]; end
    if ralpha(1) >= 1 || ralpha(1) <= 0, error('r must be in (0, 1)!'); end
    if ralpha(2) < 0, error('alpha must be >= 0!'); end

    defaults = {'A_init', [], ...
                'khl_init', true, ...
                'max_epoch', 500, ...
                'opt_tol', 1e-5, ...
                'prog_tol', 1e-9, ...
                'corrections', 100, ...
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

    [~, K] = size(X);
    w_init = (K + (N - 1) * ralpha(2)) * ones(N, 1);

    param_init = {A_init, w_init};

    % Minimize the objective function of DSF
    obj_func = @(param, X) cost_function(param, X, ralpha, o.norm_eps);
    [best_param, J] = optimize(obj_func, param_init, X, o.max_epoch, o.opt_tol, ...
        o.prog_tol, o.corrections, o.deriv_check, ...
        o.use_mex, o.verbose);
    best_A = best_param{1};
    % best_w = best_param{2};

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

function [cost, grad] = cost_function(param, X, ralpha, norm_eps)
    A = param{1}; w = param{2};
    r = ralpha(1); alpha = ralpha(2);

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

    wmask = w < 0;
    w(wmask) = 0; w = w + alpha;
    Y2 = Y2 + eps; % prevent Y2 from going to zero
    [Y2_min, minIdx] = min(Y2, [], 1);

    Y2_contra = Y2 ./ Y2_min;

    Dr2 = Y2_contra.^(r - 2);
    Dr1 = Dr2 .* Y2_contra;
    Dr = Dr1 .* Y2_contra;

    wSumDr = sum(w .* Dr, 1);
    wSumDr1 = sum(w .* Dr1, 1);

    g = wSumDr ./ wSumDr1;
    g = sum(Y2_min .* g, 2);

    dgdD = (w .* Dr2 ./ wSumDr1.^2) .* ((r - 1) .* wSumDr - r .* Y2_contra .* wSumDr1);
    dgdw = Y2_min .* (Dr .* wSumDr1 - Dr1 .* wSumDr) ./ (wSumDr1.^2);
    dgdw(wmask) = 0;

    gradw = mean(dgdw, 2);

    cost = (1 / K) * g;
    gradD = 2 * (1 / K) * X * (angle_ .* dgdD)';
    gradD = cast(gradD, class(A));

    % Apply the chain rule through column norm normalization
    gradD = gradD - bsxfun(@times, Anorm, sum(real(conj(Anorm) .* gradD), 1));
    gradD = bsxfun(@rdivide, gradD, column_norm_A);

    % Apply the chain rule through row-wise decoupling
    Sinv = diag(1 ./ diag(S));
    C = -(Sinv * U' * gradD * V) ./ (bsxfun(@plus, diag(S), diag(S)'));
    gradD = U * (C' + C) * S * V' + U * Sinv * U' * gradD;

    grad = {gradD, gradw};
end
