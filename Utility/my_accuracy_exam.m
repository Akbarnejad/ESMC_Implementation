function [accuracy] = my_accuracy_exam(Y, Y_hat)
% example accuracy

accuracy = sum(Y & Y_hat, 2) ./ sum(Y | Y_hat, 2);
accuracy(isinf(accuracy)) = 0.5;
accuracy(isnan(accuracy)) = 0.5;

end