function [true_class_probs,all_probs] = get_test_outputs_featsel(expert_params,test_data,test_targets)

test_data = [test_data ones(size(test_data,1),1)]; 

K = length(expert_params);
N = size(test_data,1);
M = size(expert_params{1}.class_w,1); 

for k = 1:K
	clust_v = expert_params{k}.clust_v;
	clust_v_featsel = expert_params{k}.v_featsel;
	all_clust_v(k,:) = clust_v .* clust_v_featsel; 
end

data_wts = compute_classifier_data_wts(test_data,all_clust_v);

summed_probs = zeros(M,N);
for k = 1:K

	w_featsel = expert_params{k}.w_featsel;
	class_w = expert_params{k}.class_w;
	w_featsel = repmat(w_featsel,size(class_w,1),1);
	class_w = class_w .* w_featsel;
 
	classifier_probs = compute_classifier_data_probs(test_data,class_w);
	
	cur_data_wts = repmat(data_wts(k,:),M,1);
	summed_probs = summed_probs + cur_data_wts.*classifier_probs;
end


% compute log likelihood on the test data
for iter = 1:N
	true_class_probs(iter) = summed_probs(test_targets(iter),iter);
end
disp('test_data_likelihood')
disp(sum(log(true_class_probs)));
all_probs = summed_probs;

