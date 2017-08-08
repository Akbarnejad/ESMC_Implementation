function [macro_F1, micro_F1, accuracy] = calc_measure(y_tt, G_tt)
y_tt = y_tt';
G_tt = G_tt';
[n2 testSize] = size(y_tt); %% 101 12914
% example based measures:

% macro F1:
% makhraj_p = sum(y_tt, 2);
% notz_p = makhraj_p ~= 0;
% makhraj_r = sum(G_tt, 2);
% notz_r = makhraj_r ~= 0;
% soorat = sum(G_tt & y_tt, 2);
% notz_indx = find(notz_p & notz_r);
% length(notz_indx) ;
% p = soorat(notz_indx) ./ makhraj_p(notz_indx);
% r = soorat(notz_indx) ./ makhraj_r(notz_indx);
% makhraj = p + r;
% notz_indx = find(makhraj ~= 0);
% soorat = 2 * p .* r;
% macro_F1 = sum(soorat(notz_indx) ./ makhraj(notz_indx)) / n2 %% .5189

% an other better way - macro F1:
TP = sum((~ xor(G_tt, y_tt)) & G_tt, 2);
v1 = TP(1:10);
TN = sum((~ xor(G_tt, y_tt)) & (~ G_tt), 2);
v2 = TN(1:10);
FP = sum(xor(G_tt, y_tt) & G_tt, 2);
v3 = FP(1:10);
FN = sum(xor(G_tt, y_tt) & (~ G_tt), 2);
v4 = FN(1:10);
makhraj = 2 * TP + FN + FP;
notz_indx = find(makhraj ~= 0);
soorat = 2 * TP;
macro_F1 = sum(soorat(notz_indx) ./ makhraj(notz_indx)) / length(notz_indx);

% micro F1
makhraj = sum(2 * TP + FN + FP);
soorat = sum(2 * TP);
micro_F1 = soorat / makhraj;

% accuracy:
makhraj = sum(G_tt | y_tt);
soorat = sum(G_tt & y_tt);
notz_indx = find(makhraj ~= 0);
length(notz_indx); %% 12822
accuracy = sum(soorat(notz_indx) ./ makhraj(notz_indx)) / length(notz_indx); %% .3011
end