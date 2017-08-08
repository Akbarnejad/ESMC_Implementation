function [ak_network] = ak_mf_variational_inference_pca_init(x,y,L,Q,alpha,M,sigma,gamma,vi_max_iters)
%
%inputs:
%   x: inputs,  a NxQ1 matrix.
%   y: outputs, a NxQL matrix.
%   L: number of layers.
%   Q: a 1xL vector, containing number of nodes in each layer.
%   alpha: a 1x(L-1) vector, containing noise of GPs of l'th layers.
%   M: a 1xL vector, containing number of sparse samples in each layer.
%   sigma: a 1x(L-1) vector, containing rbf kernel parameters of l'th layers.
%   gamma: a 1xL vector, containing variance of generating X_{el} from
%         tild{X_el}.
%   vi_max_iters: maximum number of coordinate descend iterations.
%outputs:
%   

%get some constants
[N,~] = size(x);

%find initiali values for layers, using backward PCA.
all_pca_x = cell(1,L);
all_pca_x{1,L} = y;
all_pca_x{1,1} = x;
for counter=(L-1:(-1):2)
    if(counter==L-1)
     [~,temp_pca,~] = pca(all_pca_x{1,counter+1});
     temp_pca = temp_pca(:, (1:Q(counter)));
     all_pca_x{1,counter} = temp_pca;
    else
       all_pca_x{1,counter} = randn(N,Q(counter)); 
    end
end


%sets sigma via pca initial values
%for el=2:L-1
%    sigma(el) = ak_get_proper_rbf_sigma(all_pca_x{1,el});
%end
%sigma

%data structure for all variational parameters.
all_a = cell(1,L-1);    %each element is f(Q_{el+1},M_el}
all_hasht = cell(1,L-1);%each element is f(M_{el},M_{el}), because hasht's
                        %assumed to be independent from dimension,
all_muld_var_1 = cell(1,L-1);   %each element is M_{el}xM_{el}
all_muld_var_2 = cell(1,L-1);   %each element is f(M_{el},Q_{el+1})
all_etha = cell(1,L);   %each element is NxQ_{el}, first and last elemens are fixed.
all_z = cell(1,L-1);    %each element is M_{el}xQ_{el}.
all_z_index = cell(1,L-1); %each element is a 1xM_{el} vector, containing sample indices of instances.


%randomly initialize variational parameters.
for el=1:(L-1)
   all_a{1,el} = 1*randn(Q(1,el+1),M(1,el));
end
for el=1:(L-1)
   all_hasht{1,el} = 1*eye(M(1,el)); 
end
for el=1:(L-1)
   all_muld_var_1{1,el} = 1*eye(M(1,el)); 
end
for el=1:(L-1)
   all_muld_var_2{1,el} = 100*randn(M(el) , Q(el+1)); 
end
for el=1:L  %initialize ethas
   %first and last etha is fixed and observed.
   if(el==1)
       all_etha{1,el} = x;
   elseif(el==L)
       all_etha{1,el} = y;
   else
       all_etha{1,el} = all_pca_x{1,el};
   end
end
for el=1:(L-1)  %initilize z's.
    %initilize first z by subsampling from x,
    if(el==1)
       %for first layer.
       if(M(1)<N) 
            temp = randperm(N,M(1));
            all_z_index{1,el} = temp;
            pca_el = all_pca_x{1,el};
            all_z{1,el} = pca_el(temp,:);
       else
           all_z_index{1,el} = (1:N);
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
            %if(iter>3)
                temp = ak_fast_cross_rbf_kernel(all_etha{el-1},all_z{el-1},sigma(el-1));
                all_etha{el} = temp * all_muld_var_1{el-1} * all_muld_var_2{el-1};
            %end
        end
        %end of updating etha(el)
        
        
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


end

