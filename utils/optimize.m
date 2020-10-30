%  Minimization of real-valued functions of complex-valued arguments or 
%  real-valued functions of mixed complex-valued and real-valued arguments 
%  using real-valued LBFGS toolbox.
%
%  minFunc should be in MATLAB's path.
%
% Notes
% -----
%   This program is distributed only for non-commercial research use
%   at universities, academic institutions, or non-profit organizations. 
%   Commercial uses and redistribution are not permitted without
%   permission from the authors. 
%
%   Please kindly cite the following paper if you use or modify any 
%   part of this program:
%
%   A. H. T. Nguyen, V. G. Reju, and A. W. H. Khong, “Directional Sparse Filtering 
%   for Blind Estimation of Under-determined Complex-valued Mixing Matrices,” IEEE 
%   Transactions on Signal Processing, vol. 68, pp. 1990-2003, Mar. 2020.
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.                                 --%
function [best_param, J] = optimize(obj_func, param_init, data, max_epoch, opt_tol,  ...
                                    prog_tol, corrections, deriv_check, use_mex, verbose)
  % - obj_func: function handle to the objective function in form of
  %     [cost, grad] = obj_func(param, data) to be minimized
  % - param_init: We support param as a n-dimentional array (ndarray)
  %     or a nested cell where leaf components are ndarray
  %     Eg, {A1, {A2; b2}; {{A3, b3}, A2}, {{}}} where Ai
  %     and bi are complex/real matrix.
  % - data: M by K data matrix, each column is a sample
  % NOTE: When dimension mismatching errors occur, make sure the objective
  %   function return real gradient for real input param

  % Flatten the param input and the returned gradient into vectors.
  % When a component of param is complex ndarray, its flatten vector
  % consists of the real part followed by the imaginary part, as such the param and
  % its gradient are real vectors. E.g., param = [1+2i, 3+4i] becomes [1; 3; 2; 4].
  % We allow param's components (when param is nested cell) to be mixed between real and
  % complex matrix, e.g., param = {[1+2i, 3+4i], [5, 6]} => [1; 3; 2; 4; 5; 6].
  % Note that, updating a real-valued flatten representation of a complex matrix
  % by real-valued flatten representation of gradient
  % is equivalent to updating the original matrix  with the complex gradient directly
  % See e.g. Sorber et al. 2012, "UNCONSTRAINED OPTIMIZATION OF REAL FUNCTIONS IN COMPLEX 
  % VARIABLES".
  shape = get_shape(param_init);
  param_init = flatten(param_init, shape);
  obj_func = @(param_flat, data) flatten_obj_func(obj_func, param_flat, data, shape);

  if deriv_check == 1
    fast_derivative_check(obj_func, param_init, data);
  elseif deriv_check == 2
    slow_derivative_check(obj_func, param_init, data, shape);
  end

  % Real lbfgs toolbox, we treats real/imaginary parts as separate variables
  if ~verbose, opts.Display = 'Off'; end
  opts.maxIter = max_epoch;
  opts.optTol = opt_tol;
  opts.progTol = prog_tol;
  opts.corrections = corrections;
  % opts.DerivativeCheck = 'on';
  % opts.numDiff = 0;
  opts.useMex = 1;
  if ~use_mex, opts.useMex = 0; end
  f_df = @(param) obj_func(param, data);
  [best_param, ~, ~, J] = minFunc(f_df, param_init, opts);
  J = J.trace.fval(1:end-1);

  % Restore the original shape of the parameter.
  best_param = unflatten(best_param, shape);
end

function [cost, grad_flat] = flatten_obj_func(f_df, param_flat, data, shape)
  param = unflatten(param_flat, shape);
  [cost, grad] = f_df(param, data);
  grad_flat= flatten(grad, shape);
end

function shape = get_shape(param)
  dims = size(param);
  % if prod(dims) == 1, dims = 1; end

  if ~iscell(param)
    shape.dims = dims;
    if isreal(param)
      shape.size = prod(dims);
    else
      shape.size = -2*prod(dims); % Using -negative to mark complex leaf components
    end
  else
    shape.dims = cell(dims);
    shape.size = 0;
    for i = 1:prod(dims)
      shape.dims{i} = get_shape(param{i});
      shape.size = shape.size + abs(shape.dims{i}.size);
    end
  end
end

function param_flat = flatten(param, shape)
  if ~iscell(shape.dims)
    if (shape.size < 0) % Complex
      param_flat = [real(param(:)); imag(param(:))];
    else
      param_flat = param(:);
    end
  else
    param_flat = NaN(abs(shape.size), 1);
    sidx = 1;
    for i = 1:numel(shape.dims)
      eidx = sidx+abs(shape.dims{i}.size)-1;
      param_flat(sidx:eidx) = flatten(param{i}, shape.dims{i});
      sidx = eidx+1;
    end
  end
end

function x = unflatten(param_flat, shape)
  if ~iscell(shape.dims)
    if shape.size < 0 % Complex
      param_flat = complex(param_flat(1:length(param_flat)/2), ...
                            param_flat(1+length(param_flat)/2:end));
    end
    x = reshape(param_flat, shape.dims);
  else
    x = cell(size(shape.dims));
    sidx = 1;
    for i = 1:numel(shape.dims)
      eidx = sidx+abs(shape.dims{i}.size)-1;
      x{i} = unflatten(param_flat(sidx:eidx), shape.dims{i});
      sidx = eidx+1;
    end
  end
end

function fast_derivative_check(obj_func, param, data)
  d = sign(randn(length(param), 1));
  [~, g] = obj_func(param, data);
  gtd = g'*d;

  e = sqrt(eps(class(param)));
  gtd2 = (obj_func(param+e*d, data) - obj_func(param-e*d, data))/(2*e);

  fprintf(['Relative difference between analytical and numerical' ...
           ' directional-derivative is %e\n'], abs(gtd - gtd2)/(abs(gtd)+abs(gtd2)));
end

function slow_derivative_check(obj_func, param, data, shape)
  e = sqrt(eps(class(param)));
  [~, grad] = obj_func(param, data);
  grad2 = grad*0;
  for i = 1:length(param)
      param_h = param; param_h(i) = param_h(i) + e;
      param_l = param; param_l(i) = param_l(i) - e;
      grad2(i) = (obj_func(param_h, data) - ...
                      obj_func(param_l, data))/(2*e);
  end
  fprintf(['Relative difference between analytical' ...
         ' and numerical derivative is %e\n'], ...
         norm(grad(:)-grad2(:))/(norm(grad(:))+norm(grad2(:))));
  g1 = unflatten(grad, shape);
  g2 = unflatten(grad2, shape);
  keyboard;
end
