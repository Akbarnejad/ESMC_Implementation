function [ output ] = ak_matrix_catrnd(p)
%%
%input:
%   p: a AxBxC matrix
%output:
%   in each floor of p, considers each column of that floor as parameters
%   of a categorical distribution
%   output is a BxC matrix, containing samples of categoricals
%   Automatically sums categorical parameters to one

[A,B,C] = size(p);

%sum paramters to one
temp = sum(p,1);   %temp is 1xBxC
temp = repmat(temp , [A 1 1]);
p = p./temp;



%cumulate columns of p in each floor
p = cumsum(p);

%generate samples in [0,1]
samples = rand(1,B,C);
samples = repmat(samples , [A 1 1]);


%do final operations
p = samples-p;
p = sign(p);
p(p==0) = 1;
p = sum(p); %p is now 1xBxC
p = (p+A)/2 + 1;
if(sum(sum(find(p>A))) > 0)
   error('error in ak_matrix_catrnd'); 
end
output = zeros(B,C);
output(:,:) = p(1,:,:);


end

