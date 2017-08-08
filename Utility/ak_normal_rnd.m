function [ output ] = ak_normal_rnd(mu , sigma)
%generates a random number from N(mu,sigma)

z = randn(1,numel(mu));
A = chol(sigma);
output = mu'+z*A;
output = output';

end

