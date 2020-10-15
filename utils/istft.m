% ISTFT with support of variable overlap, window, and zero-padding
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
%
% References
% ----------
%     stft_multi.m (SISEC2008), Copyright 2008 Emmanuel Vincent
function [x] = istft(X, n_sampl, NFFT, n_overlap, window, pad_to)
  if nargin < 2, error('\nNot enough argument.\n'); end
  if nargin < 3, NFFT = 256; end
  if nargin < 4, n_overlap = 0; end
  if nargin < 5, window = sin((.5:NFFT-.5)/NFFT*pi); end 
  if nargin < 6, pad_to = NFFT; end
  if NFFT ~= length(window), error('Length of window should be equal to NFFT'); end
  if n_overlap >= NFFT, error('n_overlap must be smaller than NFFT'); end
  [n_freq, n_frame, n_src, n_chan] = size(X);
  if 2*(n_freq-1) ~= pad_to, error('\nInvalid dimension, check pad_to!'); end

  window = window(:)';

  step = NFFT - n_overlap;
  % Calculating scaling vector which is later used it for make sure COLA holden
  swin = zeros(1, n_frame*step+n_overlap);
  for t = 0:n_frame-1
    swin(t*step+1:t*step+NFFT) = swin(t*step+1:t*step+NFFT) + window.^2;
  end
  swin=sqrt(swin/NFFT);

  x = zeros(n_src, n_frame*step+n_overlap, n_chan);
  for i = 1:n_chan
    for j = 1:n_src
      for t = 0:n_frame-1
        % Restore the upper part of the FFT
        fframe = [X(:, t+1, j, i); conj(X(pad_to/2:-1:2, t+1, j, i))];
        frame = real(ifft(fframe, pad_to, 1));
        % Trimming down the zeropadding of fft
        frame = frame(1:NFFT).';
        x(j, t*step+1:t*step+NFFT, i) = x(j, t*step+1:t*step+NFFT, i) + frame.*window./swin(t*step+1:t*step+NFFT);
      end
    end
  end

  % Truncation
  x = x(:, n_overlap/2+1:n_overlap/2+n_sampl, :);
end





