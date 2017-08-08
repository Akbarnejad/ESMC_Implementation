function [output] = ak_pass_data_to_model(ak_network,test_data)
%
%inputs:
%       ak_network: a deep gaussian network, which is output of
%                   ak_mf_variational_inference.
%       test_data: a (test_size)xF matrix, containing input data in it's
%                  rows
%output: a (test_size)x(Q_L) matrix, containing representations of input data in last layer.

%get network parameters.
all_a = ak_network.all_a;
all_hasht = ak_network.all_hasht;
all_etha = ak_network.all_etha;
all_muld_var_1 = ak_network.all_muld_var_1;
all_muld_var_2 = ak_network.all_muld_var_2;
all_z = ak_network.all_z;
L = ak_network.L;
Q = ak_network.Q;
M = ak_network.M;
sigma = ak_network.sigma;
alpha = ak_network.alpha;
gamma = ak_network.gamma;
[T,~] = size(test_data);


%make output
output = cell(1,L);
output{1} = test_data;  %first layer representations are test_data.
for el=1:(L-1)
   
    temp = ak_fast_cross_rbf_kernel(output{el},all_z{el},sigma(el));
    output{el+1} = temp * all_muld_var_1{el}*all_muld_var_2{el}; 
    
end

end

