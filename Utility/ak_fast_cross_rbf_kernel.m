function [ output ] = ak_fast_cross_rbf_kernel(A,B,sigma)
%
%
%
%
%this code downloaded from:
%http://stackoverflow.com/questions/23911670/efficiently-compute-pairwise-squared-euclidean-distance-in-matlab

[numA,~] = size(A);
[numB,d] = size(B);

helpA = zeros(numA,3*d);
helpB = zeros(numB,3*d);
for idx = 1:d
    helpA(:,3*idx-2:3*idx) = [ones(numA,1), -2*A(:,idx), A(:,idx).^2 ];
    helpB(:,3*idx-2:3*idx) = [B(:,idx).^2 ,    B(:,idx), ones(numB,1)];
end
distance_matrix = helpA * helpB';

%compute euclidian rbf_kernel_matrix from distance_matrix
output = (-1/(2*(sigma^2))) * distance_matrix;
output = exp(output);

end

