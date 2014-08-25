function [output] = get_results_path_speaker_wise_splits()

all_data = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/all.data');
all_targets = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/all.targets');
split_ids = load('/home/rcf-proj/mv/guptarah/MoE/data/pathology_data/data_to_use/split_info');

unique_ids = unique(split_ids)';

all_true_class_probs_test = [];
all_test_targets = [];
all_true_class_probs_dev = [];
all_dev_targets = [];
for cur_id = unique_ids

	disp('id num:');
	disp(cur_id);

	% train data split
	train_data = all_data;
	train_data(find(or(split_ids==cur_id,split_ids==mod(cur_id,39)+1)),:) = [];
	train_targets = all_targets;
	train_targets(find(or(split_ids==cur_id,split_ids==mod(cur_id,39)+1)),:) = [];

	% test data split
	test_data = all_data(find(split_ids==cur_id),:);
	test_targets =  all_targets(find(split_ids==cur_id),:);

	% dev data split
	dev_data = all_data(find(split_ids==mod(cur_id,39)+1),:);
	dev_targets = all_targets(find(split_ids==mod(cur_id,39)+1),:);

	% normalize the data
	[norm_data,mu,sigma] = zscore(train_data);
	norm_test_data = (test_data - repmat(mu,size(test_data,1),1))./ repmat(sigma,size(test_data,1),1);
	norm_dev_data = (dev_data - repmat(mu,size(dev_data,1),1))./ repmat(sigma,size(dev_data,1),1);

	num_experts = 1;
	max_iter = 300;
	expert_params = moe_train(norm_data,train_targets,num_experts,max_iter);
	[test_true_class_probs,test_all_probs] = get_test_outputs(expert_params,norm_test_data,test_targets);  
	[dev_true_class_probs,dev_all_probs] = get_test_outputs(expert_params,norm_dev_data,dev_targets);

	disp('dev accuracy');
	disp(mean(dev_true_class_probs > .5));
	
	disp('test accuracy');
	disp(mean(test_true_class_probs > .5));

	all_true_class_probs_test = [all_true_class_probs_test test_true_class_probs];
	all_test_targets = [all_test_targets test_targets']; 

	all_true_class_probs_dev = [all_true_class_probs_dev dev_true_class_probs];
        all_dev_targets = [all_dev_targets dev_targets'];
		
end

% get accuracy on test set
disp('TEST SET NUMBERS');
prob_thresh1 = 0.5;
prob_thresh2 = 1-prob_thresh1;

% get classwise accuracies
disp('class 1 accuracy');
disp(mean(all_true_class_probs_test(find(all_test_targets == 1))>prob_thresh1));

disp('class 2 accuracy');
disp(mean(all_true_class_probs_test(find(all_test_targets == 2))>prob_thresh2));

disp('unweighted accuracy');
disp(mean([mean(all_true_class_probs_test(find(all_test_targets == 1))>prob_thresh1),mean(all_true_class_probs_test(find(all_test_targets == 2))>prob_thresh2)]));

disp('Accuracy');
concat_results = [(all_true_class_probs_test(find(all_test_targets == 1))>prob_thresh1),(all_true_class_probs_test(find(all_test_targets == 2))>prob_thresh2)];
disp(mean(concat_results));


test_output = [all_true_class_probs_test;all_test_targets];


% get accuracy on dev set
disp('DEV SET NUMBERS');
prob_thresh1 = 0.5;
prob_thresh2 = 1-prob_thresh1;

% get classwise accuracies
disp('class 1 accuracy');
disp(mean(all_true_class_probs_dev(find(all_dev_targets == 1))>prob_thresh1));

disp('class 2 accuracy');
disp(mean(all_true_class_probs_dev(find(all_dev_targets == 2))>prob_thresh2));

disp('unweighted accuracy');
disp(mean([mean(all_true_class_probs_dev(find(all_dev_targets == 1))>prob_thresh1),mean(all_true_class_probs_dev(find(all_dev_targets == 2))>prob_thresh2)]));

disp('Accuracy');
concat_results = [(all_true_class_probs_dev(find(all_dev_targets == 1))>prob_thresh1),(all_true_class_probs_dev(find(all_dev_targets == 2))>prob_thresh2)];
disp(mean(concat_results));


dev_output = [all_true_class_probs_dev;all_dev_targets];

output{1} = test_output;
output{2} = dev_output;

save('output_vanilla_moe','output');
