%   Written by Anh H. T. Nguyen (nguyenha001@e.ntu.edu.sg)
%   Copyright (c) 2020, Anh H. T. Nguyen. All rights reserved.
function o = parse_options(o, defaults)
  for i = 1:2:length(defaults)
    if isfield(o, defaults{i})
      if isempty(o.(defaults{i}))
        o.(defaults{i}) = defaults{i+1};
      end
    else
      o.(defaults{i}) = defaults{i+1};
    end
  end
end
