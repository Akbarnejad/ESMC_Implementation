function [auc,false_possitives,true_possitives] = ak_auc_tp_fp_diffrent_ks(y_hat,Y_test)
%
%

%get some constants
[N,K] = size(Y_test);

%make x,y in plots
tps = zeros(1,K);
fps = zeros(1,K);
[~,si] = sort(y_hat,2,'descend');
for k=1:K
    
    %GUI
    %disp(['iteration: ' num2str(k) ' of ' num2str(K)]);
    
    %make computed_y
    computed_y = y_hat;
    computed_y(:) = 0;
    for n=1:N
        %[~,si] = sort(y_hat(n,:),'descend');
        computed_y(n,si(n,1:k)) = 1;
    end
    tp = sum(sum(Y_test.*computed_y));
    fp = sum(sum((1-Y_test).*computed_y));
    tn = sum(sum((1-Y_test).*(1-computed_y)));
    fn = sum(sum(Y_test.*(1-computed_y)));
    tps(1,k) = tp/(tp+fn);
    fps(1,k) = fp/(fp+tn);
end

%return results
false_possitives = fps;
true_possitives  = tps;
auc =ak_general_get_area_under_plot(false_possitives,true_possitives);

end

