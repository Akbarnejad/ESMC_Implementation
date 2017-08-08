
%SETTINGS
DATASET = 'corel5k';


%load data
data = load(['Datasets/' DATASET '.mat']);

%make parameters
parameters = esmc_default_parameters_with_experts(data);

%run esmc
[y_hat,auc,negated_coverage] = ESMC(data,parameters);
