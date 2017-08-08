function [output] = ak_get_scatter_matrix( input )
%
%input is a NxF matrix.
%output is \sum_n{ input(n,:)' * input(n,:)}

[N,F] = size(input);
mean = sum(input,1)/N;
output = cov(input)*(N-1)+(mean'*mean)*(N);

end

