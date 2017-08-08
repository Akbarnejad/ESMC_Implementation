function [ output ] = ak_fast_type2_kernel(A_in,B_in,teta,sigma,dim1)
%
%A: a NxF matrix.
%B: a MxF matrix
%teta: parameters, containing teta.a and teta.b.

%dim1: dimensionality of the first part.
%
%this code downloaded from:
%http://stackoverflow.com/questions/23911670/efficiently-compute-pairwise-squared-euclidean-distance-in-matlab

%make copies from input matrices.
A = A_in;
B = B_in;

%modify A and B, so that type2 kernel would be computed.
[~,F] = size(A);
A(:,(1:dim1)) = A(:,(1:dim1)) * ((teta.a)^2);
B(:,(1:dim1)) = B(:,(1:dim1)) * ((teta.a)^2);
A(:,((dim1+1):F)) = A(:,((dim1+1):F)) * ((teta.b)^2);
B(:,((dim1+1):F)) = B(:,((dim1+1):F)) * ((teta.b)^2);

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

