function [ output ] = ak_zero_mean(input)
%
%input: a matrix containing samples in it's rows, a NxF matrix.


[N,~] = size(input);
avg = sum(input,1)/N;
output = input;
output = output - repmat(avg,N,1);




end

