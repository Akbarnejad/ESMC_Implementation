function [output] = ak_get_proper_rbf_sigma(input)
%
%
%input:  a matrix of size NxF.
%output: proper rbf sigma parameter for these N data.

%get pairwise distance
distance_matrix = ak_fast_get_pairwise_distance(input,input);

output = median(distance_matrix(:));


end

