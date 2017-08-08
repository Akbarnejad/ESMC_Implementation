function [ output ] = ak_fast_get_pairwise_distance(A,B)
%
%
%input:  samples, a matrix of size NxF.
%output: proper rbf sigma for these data. 

[numA,~] = size(A);
[numB,d] = size(B);

helpA = zeros(numA,3*d);
helpB = zeros(numB,3*d);
for idx = 1:d
    helpA(:,3*idx-2:3*idx) = [ones(numA,1), -2*A(:,idx), A(:,idx).^2 ];
    helpB(:,3*idx-2:3*idx) = [B(:,idx).^2 ,    B(:,idx), ones(numB,1)];
end
output = helpA * helpB';
output = sqrt(output);


end

