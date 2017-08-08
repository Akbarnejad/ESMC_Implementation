function [ ak_network ] = ak_semisupervied_mf_variational_inference_with_experts(x,y,L,Q,alpha,M,sigma,sigma_pretrain,gamma,B,landa,ak_const,vi_max_iters_initialization,vi_max_iters)
%
%inputs:
%   x: inputs,  a (N+I)xQ1 matrix containing both labeled and unlabeled
%      data. Note that unlabeled data are in bigger indices.
%   y: outputs, a NxQL (or NxK)matrix.
%   L: number of layers.
%   Q: a 1xL vector, containing number of nodes in each layer.
%   alpha: a 1x(L-1) vector, containing noise of GPs of l'th layers.
%   M: a 1xL vector, containing number of sparse samples in each layer.
%   sigma: a 1x(L-1) vector, containing rbf kernel parameters of l'th layers.
%   sigma_pretrain: a 1x(L-1) vector, containing rbf kernel for
%                   pretraining.
%   gamma: a 1xL vector, containing variance of generating X_{el} from
%         tild{X_el}.
%   B: number of experts, a scalar.
%   landa: scale factor of sigmoid function, a scalar.
%   ak_const: starting point in last layer is +- this constant.
%   vi_max_iters_initialization: number of VI iterations to find initial
%                                eta for unlabeled data.
%   vi_max_iters: maximum number of coordinate descend iterations.
%outputs:
%

%get some constants
[N,K] = size(y);
I = size(x,1)-size(y,1);

%separating labeled and unlabeled data.
x_labeled   = x((1:N),:);
x_unlabeled = x((N+1:N+I) , :);


%find initiali values for layers, using backward PCA.
f = @(t)(tanh(t./2)./(4.*t));

all_pca_x = cell(1,L);
all_pca_x{1,L} = (y-0.5)*2*ak_const;
all_pca_x{1,1} = x;
for counter=(L-1:(-1):2)
    [~,temp_pca,~] = pca(all_pca_x{1,counter+1});
    
    try   %exception happens when y is low rank, so Q(counter)>size(temp_pca)
        temp_pca = temp_pca(:, (1:Q(counter)));
    catch
        temp_pca = [temp_pca zeros(N, (Q(counter)-size(temp_pca,2)))];
        warning(['Y_train for labeled data is low rank, so pca on it has lower dimension than ' num2str(Q(counter)) '.']);
    end
    all_pca_x{1,counter} = temp_pca;
end


%data structure for all variational parameters.
all_a = cell(1,L-1);    %each element is f(Q_{el+1},M_el}
all_hasht = cell(1,L-1);%each element is f(M_{el},M_{el}), because hasht's
                        %assumed to be independent from dimension,
all_muld_var_1 = cell(1,L-1);   %each element is M_{el}xM_{el}
all_muld_var_2 = cell(1,L-1);   %each element is f(M_{el},Q_{el+1})
all_etha = cell(1,L);   %each element is (N+I)xQ_{el}, first and last elemens are fixed.
all_z = cell(1,L-1);    %each element is M_{el}xQ_{el}.
all_z_index = cell(1,L-1); %each element is a 1xM_{el} vector, containing sample indices of instances.
theta = abs(randn(N,K)+1);     %diag of fisher information matrix of X_L.
p = repmat(y,[1 1 B]);       %bernouli parameters of experts.
kisi = randn(N,K);


%randomly initialize variational parameters.
for el=1:(L-1)
   all_a{1,el} = 1*randn(Q(1,el+1),M(1,el));
end
for el=1:(L-1)
   all_hasht{1,el} = 100*eye(M(1,el)); 
end
for el=1:(L-1)
   all_muld_var_1{1,el} = 1*eye(M(1,el)); 
end
for el=1:(L-1)
   all_muld_var_2{1,el} = 100*randn(M(el) , Q(el+1)); 
end


%run mf-variational inference in supervised setting, to find initial value
%   for eta of unlabeled data.
[ak_network_initialization] = ak_mf_variational_inference_with_experts(x((1:N) , :),y,L,...
                                                                                Q,alpha,M,...
                                                                                sigma_pretrain,gamma,...
                                                                                B,landa,...
                                                                                ak_const,vi_max_iters_initialization);
x_unlabeled_init = ak_pass_data_to_model(ak_network_initialization,x_unlabeled);
%save('x_unlabeld_init.mat', 'x_unlabeled_init');

%initialize etas.
for el=1:(L)
   %first and last etha is fixed and observed(for labeled data)
   if(el==1)
       all_etha{1,el} = x;
   elseif(el==L)
       temp = zeros(N,K);
       temp(y==0) = -ak_const;
       temp(y==1) =  ak_const;
       temp_unlabeled = x_unlabeled_init{1,el};         %initial value for unlabeled data.
       temp = [temp ; temp_unlabeled];
       all_etha{1,el} = temp;
       
   else
       temp_unlabeled = x_unlabeled_init{1,el};   %initial valued for eta of unlabeled data.   
       all_etha{1,el} = [all_pca_x{1,el} ; temp_unlabeled];
   end
end
for el=1:(L-1)  %initilize z's.
    %initilize first z by subsampling from x,
    if(el==1)
       %for first layer.
       if(M(1)<(N+I)) 
            temp = randperm(N+I,M(1));%---override 115->118
            all_z_index{1,el} = temp;
            pca_el = x;
            all_z{1,el} = pca_el(temp,:);
       else
           all_z_index{1,el} = (1:(N+I));
           pca_el = all_pca_x{1,el};
           all_z{1,el} = pca_el;
       end
    else
        %for second,... layer.
        temp = randperm(N,M(el));
        all_z_index{1,el} = temp;
        pca_el = all_pca_x{1,el};
        all_z{1,el} = pca_el(temp,:);
    end
end
nu = theta .* (all_etha{1,L}((1:N) , :));        %fisher vectors of X_L.
%parameters nu, theta and p are initialized when they defined

%------------------Update Variational Parameters---------------------------
iter = 1;
all_converged = false;
while(all_converged == false)
    
    %this case is for debug
    if(vi_max_iters==0)
        break;
    end
    
    
    
    %for debug
    disp(['********iteration: ' num2str(iter) ]);
    iter = iter + 1;
    %tic;
    
    
    
    
    
    %update parameters layer by layer
    for el=(1:1:L-1)
        
        disp(['              layer: ' num2str(el)]);
        
        %-------------------If el>1, Update ehta(el)-----------------------
        if(el>1)
            temp = ak_fast_cross_rbf_kernel(all_etha{el-1},all_z{el-1},sigma(el-1));
            all_etha{el} = temp * all_muld_var_1{el-1} * all_muld_var_2{el-1};
        end
        %end of updating etha(el)
        %NOTE: etha_L is updated just after nu and theta are updated.
        
        
        %-----------------Update A_{el} and HASHT_{el}---------------------
        %update a_{el}
        first_term = ak_fast_cross_rbf_kernel(all_z{el},all_z{el},sigma(el));   %first_term is now M_{el}xM_{el}.
        first_term = first_term * all_muld_var_1{el}*all_muld_var_2{el};        %first+term is now M_{el}xQ_{el+1}.
        first_term = (1/((alpha(el))^2)) * (first_term');   %first_term is now Q_{el+1}xM_{el}.
        temp_1 = ak_fast_cross_rbf_kernel(all_etha{el},all_z{el},sigma(el));    %temp_1 is now NxM_{el}.
        temp_2 = ak_fast_cross_rbf_kernel(all_z{el},all_z{el},sigma(el));       %
        temp_2 = temp_2 + ((alpha(el)^2)*eye(M(el)));                            %
        temp_2 = temp_2 ^(-1);                                                  %temp_2 is now M_{el}xM_{el}.
        delta  = temp_1 * temp_2;   %delta is now NxM_{el}.
        second_term = (all_etha{el+1})' * delta;
        second_term = (1/((gamma(el+1))^2)) * second_term;  %second_term is now Q_{el+1}xM_{el}.
        all_a{el} = first_term + second_term;
        %update hasht_{el}
        first_term = ak_get_scatter_matrix(delta);
        first_term = (1/((gamma(el+1))^2)) * first_term;    %first_term is now M_{el}xM_{el}.
        all_hasht{el} = first_term + (1/((alpha(el))^2))*eye(M(el));
        
        
        
        %-----------------------Update Z_{el}------------------------------
        temp = all_etha{el};    %temp is now NxQ_{el}
        all_z{el} = temp(all_z_index{el},:);
        
        
        
        
        %-----------------Update MULD_VAR_1 and MULD_VAR_2-----------------
        %get mean u
        mean_u = zeros(Q(el+1) , M(el));
        temp = all_a{el};
        for counter=1:Q(el+1)
           [mu , ~] = ak_get_moments(temp(counter,:)',all_hasht{el}); 
           mean_u(counter,:) = mu;
        end
        %update muld_var_2
        all_muld_var_2{el} = mean_u';
        %update muld_var_1
        temp = ak_fast_cross_rbf_kernel(all_z{el},all_z{el},sigma(el));
        temp = temp + ((alpha(el))^2)*eye(M(el));
        all_muld_var_1{el} = temp ^(-1);
        
    end
    %end of updating each layer.
    
    
    %-------------------------UPDATE NU and THETA--------------------------
    first_term = all_etha{1,L-1}((1:N),:);   %first_term is now NxF
    first_term = ak_fast_cross_rbf_kernel(first_term,all_z{1,L-1},sigma(1,L-1));
                                    %first_term is now NxM_{L-1}
    second_term = ak_fast_cross_rbf_kernel(all_z{1,L-1},all_z{1,L-1},sigma(1,L-1));
    second_term = second_term + ((alpha(1,L-1))^2)*eye(M(L-1));
    second_term = (second_term^(-1));%second_term is now M_{L-1}xM_{L-1}
    %get mean_u_{L-1}
    mean_u_l_1 = zeros(M(L-1),K);
    temp = all_a{1,L-1};
    for k=1:K
        [mu , ~] = ak_get_moments(temp(k,:)',all_hasht{L-1});
        mean_u_l_1(:,k) = mu;
    end
    eta_tilda = first_term * second_term * mean_u_l_1;  %eta_tilda is now NxK
    second_term = sum(p,3); %second_term is now NxK
    nu = ((1/(gamma(L)^2))*(eta_tilda)) + (landa*second_term) - ((B*landa)/2);
    theta = (1/(gamma(L)^2)) + ((2*B*(landa^2))*(f(kisi)));
    %update etha_{L}
    first_term = 1 ./ theta;    %first_term is now NxK
    eta_last_layer = first_term .* nu;
    eta_last_layer(y==1) = 1000;
    all_etha{1,L}((1:N) , :) = eta_last_layer;
    
    
    
    
    
    %--------------------------------UPDATE P------------------------------
    %general form
    temp = all_etha{1,L}((1:N) , :);   %temp is now NxK
    temp = -landa*temp;
    temp = 1 + exp(temp);
    temp = 1 ./ temp;
    temp = repmat(temp,[1 1 B]);
    %apply first condition
    mask = repmat(y,[1 1 B]);
    mask = logical(mask);
    temp(mask) = 1;
    %apply second condition
    temp(:,:,1) = temp(:,:,1) .* y;
    p = temp;
    
    
    
    
    %------------------------------UPDATE KISI-----------------------------
    kisi = -landa * (all_etha{1,L}((1:N) , :));
    
    
    
    
    %check if converged,
    if(iter>vi_max_iters)
       all_converged = true; 
    end
    
    %toc;
end
%end of updating variational parameters.

%return a deep gaussian network.
ak_network.all_a = all_a;
ak_network.all_hasht = all_hasht;
ak_network.all_etha  = all_etha;
ak_network.all_muld_var_1 = all_muld_var_1;
ak_network.all_muld_var_2 = all_muld_var_2;
ak_network.all_z = all_z;
ak_network.L = L;
ak_network.Q = Q;
ak_network.M = M;
ak_network.sigma = sigma;
ak_network.alpha = alpha;
ak_network.gamma = gamma;
if(exist('eta_tilda'))
    ak_network.eta_tilda = eta_tilda;
end
ak_network.p = p;
ak_network.kisi = kisi;
ak_network.f = f;
ak_network.nu = nu;
ak_network.theta = theta;




end

