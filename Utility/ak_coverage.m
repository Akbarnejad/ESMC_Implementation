function [coverage] = ak_coverage(y_hat,Y_test)
%
%

%get some constants
[N,K] = size(Y_test);

%make rank matrix
rank_matrix = zeros(N,K);
[~,si] = sort(y_hat,2,'descend');
for n=1:N
    current_row = y_hat(n,:);
    %[~,si] = sort(current_row,'descend');
    rank_matrix(n , si(n,:)) = [1:1:K]; 
end

%make output
output = zeros(1,N);
for n=1:N
   temp = rank_matrix(n,:) .* Y_test(n,:);
   output(1,n) = max(max(temp));
end

%return result
coverage = sum(sum(output))/N;
coverage  = coverage-1;


end

