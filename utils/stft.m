% STFT with support of variable overlap, window, and zero-padding
%
%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
%
% References
% ----------
%     stft_multi.m (SISEC2008), Copyright 2008 Emmanuel Vincent
function [X] = stft(x, NFFT, n_overlap, window, pad_to)
  if iscolumn(x), x = x'; end
  if nargin < 2, NFFT = 256; end
  if nargin < 3, n_overlap = 0; end
  if nargin < 4, window = sin((.5:NFFT-.5)/NFFT*pi).'; end % Sine
  if nargin < 5, pad_to = NFFT; end
  if NFFT ~= length(window), error('\nLength of window should be equal to NFFT\n'); end
  if pad_to < NFFT, error('Pad_to must greater than or equal to NFFT\n'); end
  if n_overlap >= NFFT, error('n_overlap must be smaller than NFFT\n'); end

  window = window(:);
  [n_chan, n_sampl] = size(x);
  if n_chan > n_sampl, error('Input may have wrong shape!'); end
  % If n_sampl < NFFT, zero pad the input to NFFT
  if n_sampl < NFFT
    tmp = zeros(n_chan, NFFT); tmp(:, 1:n_sampl) = x; x = tmp; n_sampl = NFFT;
  end

  % Step size
  step = NFFT - n_overlap;
  n_frame = ceil(n_sampl/step);
  % Zeropadding for the last frame
  x = [x, zeros(n_chan, n_frame*step - n_sampl)];
  % Pre-processing for edges
  x = [zeros(n_chan, n_overlap/2), x, zeros(n_chan, n_overlap/2)];

  % Calculating scaling vector which is later used it for make sure COLA holden
  swin = zeros(n_frame*step+n_overlap, 1);
  for t = 0:n_frame-1
    swin(t*step+1:t*step+NFFT) = swin(t*step+1:t*step+NFFT) + window.^2;
  end
  swin=sqrt(NFFT*swin);

  % Framing, calculating FFT and take half of the spectrum
  n_freq = pad_to/2 + 1;
  X = complex(zeros(n_freq, n_frame, n_chan));
  for i = 1:n_chan
    for t = 0:n_frame-1
      frame = x(i, t*step+1:t*step+NFFT).' .* window./swin(t*step+1:t*step+NFFT);
      fframe = fft(frame, pad_to, 1);
      X(:, t+1, i) = fframe(1:n_freq);
    end
  end
end





