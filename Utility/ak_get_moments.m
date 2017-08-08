function [mu , second_moment] = ak_get_moments(v,hasht)
%%
%inputs: v and hasht, natural paramters of normal distribution
%outputs: 
%   mu: first central moment
%   second_moment: second central moment of normal distribution


sigma = hasht^(-1);
mu = sigma * v;
second_moment = sigma + (mu*mu');

end

