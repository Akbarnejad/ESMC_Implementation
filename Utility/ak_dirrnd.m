function [ output ] = ak_dirrnd(alpha)
%generates random sample from dirichlet(alpha(1xK))

k = numel(alpha)+1;

%revise format of alpha
alpha_k_1 = alpha(k-1);
new_alpha = zeros(1,k);
for counter=1:k-2
   new_alpha(counter) = alpha(counter); 
end
new_alpha(k-1) = 0.5*(alpha_k_1);
new_alpha(k)   = 0.5*(alpha_k_1);
alpha = new_alpha;


temp = zeros(1,k);
temp = gamrnd(alpha,ones(1,k));
%for counter=1:k
   %make gamma(1,alpha_i)
%   pd = makedist('Gamma' , 'a' , alpha(counter));
   
   %generate sample from gama(1,alpha_i)
%   temp(counter) = random(pd);
%end

%make output
output = zeros(1,k-1);
for counter=1:k-2
    output(counter) = temp(counter) / (sum(temp));
end
output(k-1) = (temp(k-1)+temp(k))/(sum(temp));

end

