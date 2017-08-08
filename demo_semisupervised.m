
%SETTINGS
DATASET = 'corel5k';
N = 1000;   %number of labeled instances

%load data
data = load(['Datasets/' DATASET '.mat']);
data.Y_train = data.Y_train(1:N,:); %simulating the semisupervised setting.

%make parameters
parameters = esmc_default_parameters_semisupervised(data);

%run esmc
[y_hat,auc,negated_coverage] = ESMC(data,parameters);
