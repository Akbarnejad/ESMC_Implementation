function [y_hat,auc,neg_coverage] = ESMC(data,parameters)
%

%read constants
NI = size(data.X_train,1);
N = size(data.Y_train,1);
I = NI-N;
K = size(data.Y_train,2);
F = size(data.X_train,2);
L = floor(parameters.DIM_REDUCTION_RATE*K);
if(NI>3998)
        sigma_c = ak_large_datasets_get_proper_rbf_sigma(data.X_train,3999);
else
        sigma_c = ak_large_datasets_get_proper_rbf_sigma(data.X_train,N);
end
sigma = [sigma_c , parameters.SIGMA_Z];



if(I>0)%----------------------------------- semisupervised setting -----------------------------
    
    
    %training phase
    [the_model] = ...
     ak_semisupervied_mf_variational_inference_with_experts(data.X_train,data.Y_train,...
                      3,[F L K],...
                      repmat(parameters.ALPHA,1,3),[parameters.M,parameters.M],sigma,...
                      sigma,...
                      repmat(parameters.GAMMA,1,3),parameters.B,...
                      parameters.LANDA,parameters.RHO,...
                      5,parameters.NUM_ITERS);
                  
    %test phase
    output = ak_pass_data_to_model(the_model,data.X_test);
    y_hat = output{1,3};
    
    %evaluation
    [auc,~,~] = ak_auc_tp_fp_diffrent_ks(y_hat,data.Y_test);
    coverage = ak_coverage(y_hat,data.Y_test);
    neg_coverage = -coverage;
    
elseif(parameters.B == 0)% ----------------------------------- the case B=0 -------------------------------------
    
    [the_model] = ak_mf_variational_inference_b0(data.X_train,data.Y_train,...
        3,[F L K],...
        repmat(parameters.ALPHA,1,3),[parameters.M,parameters.M],sigma,...
        repmat(parameters.GAMMA,1,3),parameters.NUM_ITERS);
    %test phase
    output = ak_pass_data_to_model(the_model,data.X_test);
    y_hat = output{1,3};
    
    %evaluation
    [auc,~,~] = ak_auc_tp_fp_diffrent_ks(y_hat,data.Y_test);
    coverage = ak_coverage(y_hat,data.Y_test);
    neg_coverage = -coverage;
    
else %------------------------------------------------------- the case B>0 ----------------------------------------
    
    
    %training phase
    [the_model] = ak_mf_variational_inference_with_experts(data.X_train,data.Y_train,...
        3,[F L K],repmat(parameters.ALPHA,1,3),...
        [parameters.M,parameters.M],sigma,repmat(parameters.GAMMA,1,3),...
        parameters.B,parameters.LANDA,parameters.RHO,parameters.NUM_ITERS);
    %test phase
    output = ak_pass_data_to_model(the_model,data.X_test);
    y_hat = output{1,3};
    
    %evaluation
    [auc,~,~] = ak_auc_tp_fp_diffrent_ks(y_hat,data.Y_test);
    coverage = ak_coverage(y_hat,data.Y_test);
    neg_coverage = -coverage;
    
end



end

