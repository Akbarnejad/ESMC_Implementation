function [ x ] = ak_bernouli_rnd_2d(p)
%%
%inputs:
%   p: a matrix
%output:
%   x: a matrix ~ Bernouli(p)

[row,col] = size(p);
x = rand(row,col)>(1-p);

end

