function [ x ] = ak_bernouli_rnd_3d(p)
%%
%inputs:
%   p: a matrix
%output:
%   x: a matrix ~ Bernouli(p)

[row,col,floor] = size(p);
x = rand(row,col,floor)>(1-p);

end

