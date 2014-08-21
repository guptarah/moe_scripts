function [all_true_class_probs all_test_targets] = get_results_path_speaker_wise_splits()

all_data = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/all.data');
all_targets = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/all.targets');
split_ids = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/split_info');

unique_ids = unique(split_ids)';

all_true_class_probs = [];
all_test_targets = [];
for cur_id = unique_ids

	disp('id num:');
	disp(cur_id);

	train_data = all_data;
	train_data(find(split_ids==cur_id),:) = [];
	train_targets = all_targets;
	train_targets(find(split_ids==cur_id),:) = [];

	test_data = all_data(find(split_ids==cur_id),:);
	test_targets =  all_targets(find(split_ids==cur_id),:);

	% normalize the data
	[norm_data,mu,sigma] = zscore(train_data);
	norm_test_data = (test_data - repmat(mu,size(test_data,1),1))./ repmat(sigma,size(test_data,1),1);

	num_experts = 4;
	max_iter = 300;
	expert_params = moe_train(norm_data,train_targets,num_experts,max_iter);
	[true_class_probs,all_probs] = get_test_outputs(expert_params,norm_test_data,test_targets);  

	disp('Accuracy');
	disp(mean(true_class_probs > .5));

	all_true_class_probs = [all_true_class_probs true_class_probs];
	all_test_targets = [all_test_targets test_targets']; 	
end

% get accuracy on test set
prob_thresh1 = 0.5;
prob_thresh2 = 1-prob_thresh1;

% get classwise accuracies
disp('class 1 accuracy');
disp(mean(all_true_class_probs(find(all_test_targets == 1))>prob_thresh1));

disp('class 2 accuracy');
disp(mean(all_true_class_probs(find(all_test_targets == 2))>prob_thresh2));

disp('unweighted accuracy');
disp(mean([mean(all_true_class_probs(find(all_test_targets == 1))>prob_thresh1),mean(all_true_class_probs(find(all_test_targets == 2))>prob_thresh2)]));

disp('Accuracy');
concat_results = [(all_true_class_probs(find(all_test_targets == 1))>prob_thresh1),(all_true_class_probs(find(all_test_targets == 2))>prob_thresh2)];
disp(mean(concat_results));


output = [all_true_class_probs;all_test_targets];
save('output_vanilla_moe','output');
