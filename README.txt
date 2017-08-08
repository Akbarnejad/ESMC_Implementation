

- This is the implementation of the paper "An Efficient (Large-scale) Semi-supervised Multi-label
  Classifier Capable of Handling Missing Labels", whose arXiv preprint is available online: https://arxiv.org/abs/1606.05725

- To run the proposed method, please use the following function:
	[y_hat,auc,negated_coverage] = ESMC(data,parameters);
  data should contain 'X_train' ((N+I)xF), 'Y_train' (NxK), 'X_test' and 'Y_test'.
  you can load default parameters via the functions "esmc_default_parameters_b0" , "esmc_default_parameters_semisupervised", and "esmc_default_parameters_with_experts".
  Sample runs are available in three files: "demo_b0.m" , "demo_with_experts.m", and "demo_semisupervised.m".
  
- Note that to obtain the results reported in the paper, you should set the parameters as explained in the paper. In fact, you should set some of the parameters using cross-validation.
  But the functions "esmc_default_parameters_b0" , "esmc_default_parameters_semisupervised", and "esmc_default_parameters_with_experts" produce parameters that are more or less suitable. 


For further information, please do not hesitate to contact the authors of the paper.
  
  
  