function [ output ,temp] = ak_test(input)
%
[N,F] = size(input);
mean = sum(input,1)/N;
%input = input = repmat(mean,N,1);
output = zeros(F,F);
tic;
for n=1:N
   output = output + ((input(n,:)')*input(n,:)); 
end
toc;

tic;
temp = cov(input)*(N-1)+(mean'*mean)*(N);
toc;

disp(max(max(temp-output)))
end

