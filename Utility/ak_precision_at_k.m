function [ output ] = ak_precision_at_k(Y_scores,Y_true,topk)
%
%inputs:
%   Y_scores: label scores, predictions of the algorithm in continous space
%   Y_true: ground-truth labels.
%   topk: precision when topk most relevant labels are selected.

[T,K] = size(Y_true);

computed_y = zeros(T,K);
for t=1:T
   %sort t'th row
   [~,si] = sort(Y_scores(t,:),'descend');
   
   %assign t'th row of computed_y
   computed_y(t , si(1:topk)) = 1;
   
end


%return output
tp = sum(sum(Y_true .* computed_y));
output = tp /(T*topk);


end

