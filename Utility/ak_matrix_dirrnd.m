function [ output ] = ak_matrix_dirrnd(alpha)
%%
%input: alpha_{NxK}, each row is parameter vector of a dirichlet distribution
%output: output_{NxK} each row sampled from a dirichlet distribution

%getting N and K
[N,K] = size(alpha);

%make modified matrix(temp)
temp = [alpha alpha(:,K)];
temp(:,K)   = temp(:,K)*0.5;
temp(:,K+1) = temp(:,K+1)*0.5;

%generate gamma samples
samples = gamrnd(temp , ones(N,K+1));

%make output using samples
s = sum(samples,2);
s = repmat(s,1,K);
output = zeros(N,K);
output(:,1:K-1) = samples(:,1:K-1);
output(:,K) = samples(:,K) + samples(:,K+1);
output = output ./ s;



end

