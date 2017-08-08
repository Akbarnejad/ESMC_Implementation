function [ output ] = esmc_default_parameters_semisupervised(data)

[N,K] = size(data.Y_train);
B = sum(sum(data.Y_train==0))/sum(sum(data.Y_train==1));
B = floor(B);
if(B>50)
   B = 50; 
end

output.B = B;
output.M = 200;
output.DIM_REDUCTION_RATE = 0.25;
output.SIGMA_Z = 1000;
output.ALPHA = 0.001;
output.GAMMA = 0.0001;
output.RHO = 100;
output.LANDA = 100000;
output.NUM_ITERS = 20;

end

