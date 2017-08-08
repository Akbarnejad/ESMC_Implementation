function [ output ] = ak_normalize_to_range(x,begin_range,end_range)
%
%inputs:
%   x: input data, a NxF matrix.
%   begin_range: begin of interval, a scalar.
%   end_range:   end of interval, a scalar.
%
%outputs:
%   output: normalized data.

[N,~] = size(x);
temp = sum(x,1);
temp = temp/N;
temp = repmat(temp,[N,1]);


output = x-temp;
output = output/max(max(output));   %output is in [0,1]
output = (output*(end_range-begin_range)) + begin_range;


end

