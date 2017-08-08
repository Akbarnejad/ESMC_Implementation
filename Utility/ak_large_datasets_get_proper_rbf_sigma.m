function [ proper_sigma ] = ak_large_datasets_get_proper_rbf_sigma(x,num_samples)
%
%inupts:
%   x: a matrix of size NxF, where N is big.
%   num_samples: number of samples from instances, to compute 
%                proper sigma.
%output: proper_sigma

[N,~] = size(x);
samples_idx = randperm(N,num_samples);
samples_of_x = x(samples_idx , :);
proper_sigma = ak_get_proper_rbf_sigma(samples_of_x);


end

