function get_results_path_predef_splits()

data = load ('../data/pathology_data/data_to_use/train.data');
test_data = load('../data/pathology_data/data_to_use/dev.data');
targets = load('../data/pathology_data/data_to_use/train.targets');
test_targets = load('../data/pathology_data/data_to_use/dev.targets');


%% flip the data split
%test_data = load ('../data/pathology_data/data_to_use/train.data');
%data = load('../data/pathology_data/data_to_use/dev.data');
%test_targets = load('../data/pathology_data/data_to_use/train.targets');
%targets = load('../data/pathology_data/data_to_use/dev.targets');


% z -normalize the data
[norm_data,mu,sigma] = zscore(data);
norm_test_data = (test_data - repmat(mu,size(test_data,1),1))./ repmat(sigma,size(test_data,1),1);

num_experts = 4;
max_iter = 200;
expert_params = moe_train(norm_data,targets,num_experts,max_iter);
% on train set itself
%[true_class_probs,all_probs] = get_test_outputs(expert_params,norm_data,targets);

[true_class_probs,all_probs] = get_test_outputs(expert_params,norm_test_data,test_targets);

% get accuracy on test set
prob_thresh1 = 0.45;
prob_thresh2 = 1-prob_thresh1;

% get classwise accuracies
disp('class 1 accuracy');
disp(mean(true_class_probs(find(test_targets == 1))>prob_thresh1));

disp('class 2 accuracy');
disp(mean(true_class_probs(find(test_targets == 2))>prob_thresh2));

disp('unweighted accuracy');
disp(mean([mean(true_class_probs(find(test_targets == 1))>prob_thresh1),mean(true_class_probs(find(test_targets == 2))>prob_thresh2)]));

disp('Accuracy');
concat_results = [(true_class_probs(find(test_targets == 1))>prob_thresh1),(true_class_probs(find(test_targets == 2))>prob_thresh2)];
disp(mean(concat_results));
