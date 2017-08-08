function[Gtest] = FaIE_test(Xtrain, Xtest, Ctrain, Ytrain, D, eps, eps_thre, sigm)
% Ctest = OVA_linreg(Ctrain, Xtrain, Xtest, eps);

[~, Ctest] = km_krr(Xtrain, Ctrain, 'gauss', sigm, eps, Xtest);

param = .01;

% ------------ obtain Gtest ----------------
func = sigmoid(Ctest * D, param);
Gtest = func;

% ---- obtain Gtrain, then obtain thresh_train & thresh_test
func = sigmoid(Ctrain * D, param);
Gtrain = func;
Gtrain_sort = sort(Gtrain, 2, 'descend');
[~, label_num] = size(D);
[train_num, ~] = size(Ctrain);
Gtrain_sort = [max(Gtrain_sort, [ ], 2) + .01, Gtrain_sort, min(Gtrain_sort, [ ], 2) - .01];  
thre_mat = zeros(train_num, label_num + 1);
for j = 1 : label_num + 1,
    thre_mat(:, j) = (Gtrain_sort(:, j) + Gtrain_sort(:, j + 1)) / 2;
end
F1_mat = zeros(train_num, label_num + 1);
for j = 1 : label_num + 1,
    thre_comp = repmat(thre_mat(:, j), 1, label_num);
    Ytrain_hat = Gtrain > thre_comp;
    F1_mat(:, j) = my_accuracy_exam(Ytrain, Ytrain_hat);
end
[~, thre_indx] = max(F1_mat, [ ], 2);
thresh_train = zeros(train_num, 1);
for i = 1 : train_num,
    thresh_train(i) = thre_mat(i, thre_indx(i));
end

% thresh_test = OVA_linreg(thresh_train, Xtrain, Xtest, eps_thre);  %%%%%%%%%%% cross validation is required
[~, thresh_test] = km_krr(Xtrain, thresh_train, 'gauss', sigm, eps_thre, Xtest);%%%%%% cross validation is required


% obtain zero-one Gtest
comp_thresh_test = repmat(thresh_test, 1, label_num);
Gtest = Gtest > comp_thresh_test;

%Gtest = Gtest > 0.5;


end