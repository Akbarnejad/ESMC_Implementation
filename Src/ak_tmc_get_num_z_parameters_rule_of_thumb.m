function [ output ] = ak_tmc_get_num_z_parameters_rule_of_thumb(number_of_instances)
%
%

%if N<10000, it is a small dataset.
%if 1000<N<20000 it is a medium-scale dataset
%if 20000<N, it is a large-sacle dataset.
if(number_of_instances<500)
    output = 100;
elseif(number_of_instances<10000)
    output = 500;%---oldfloor(0.05*number_of_instances);
elseif(number_of_instances < 20000)
    output = 500;%---oldfloor(0.01*number_of_instances);
else
    output = 500;%---old400;
end


end

