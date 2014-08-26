function [expert_params,all_likelihood] = moe_train(data,targets,num_experts,max_iter)

% data : each row contains one instance
% targets : discrete values 1 to M

if nargin < 4
	max_iter = 10;
end

data = [data ones(size(data,1),1)];

N = size(data,1); % num instances
D = size(data,2); % data dim
M = length(unique(targets)); % num classes
K = num_experts;

% obtain 1 in K encoding
targets_1inK = obt_1inK(targets,M,N);

% initialize parameters
expert_params = cell(1,K);

resp_pattern = .49*ones(1,N);
resp_pattern(1:K:N) = .51;

%resp_pattern = rand(1,N);

for k = 1:K
	
	expert_params{k}.class_w = ones(M,D);
	expert_params{k}.resp = (circshift(resp_pattern',k-1))';
	expert_params{k}.clust_v = ones(1,D);		
	expert_params{k}.data_probs = 1/M*ones(M,N); % a matrix where each row (corresponding to a class) contains data point probs 
	expert_params{k}.data_wts = ones(1,N); % contains the wt to multiply to above probs to find final prob
end


% learn the parameters
eta_factor = .2;
iter_counter = 0;
all_likelihood = zeros(max_iter-1,1);
while iter_counter < max_iter
	iter_counter = iter_counter + 1;

	% E step: COMPUTING RESPONSIBILITIES 
	if iter_counter > 1  % not doing it for first iter
		% compute denom for resp
		denom = zeros(1,N);
		for k = 1:K
			assigned_probs =  expert_params{k}.data_probs;
	                data_wts = expert_params{k}.data_wts;
	                true_class_probs = zeros(1,N);
	                for iter = 1:N
	                       true_class_probs(iter) = assigned_probs(targets(iter),iter);
	                end
			denom = denom + (true_class_probs.*data_wts);
		end
	
		%disp('data likelihood');
		%disp(sum(log(denom)));
		all_likelihood(iter_counter-1) = sum(log(denom));
	
		for k = 1:K
			% find responsibilities
			assigned_probs =  expert_params{k}.data_probs;
			data_wts = expert_params{k}.data_wts;
			true_class_probs = zeros(1,N);	
			for iter = 1:N
				true_class_probs(iter) = assigned_probs(targets(iter),iter);	
			end
			numer = true_class_probs.*data_wts;
			expert_params{k}.resp = numer./denom;
		end
	end

	% M step: COMPUTING THE CLASSIFIER PARAMETERS
	for k = 1:K
		% computing the classifier weights w
		data_probs = expert_params{k}.data_probs;
		resp = expert_params{k}.resp;
		cur_w = expert_params{k}.class_w;

		eta1 = eta_factor^(floor(max_iter/(max_iter-iter_counter))); % set eta
		[new_w] = update_w(data,data_probs,resp,cur_w,targets_1inK,eta1);	
		expert_params{k}.class_w = new_w;		

		% computing the classifier weights v
		cur_v = expert_params{k}.clust_v;
		eta2 = eta_factor^(floor(max_iter/(max_iter-iter_counter)));	
		[new_v] = update_v(data,cur_v,resp,eta2);
		expert_params{k}.clust_v = new_v;		

	end

	% UPDATE DATA PROBS & DATA WTS
	for k = 1:K
		class_w = expert_params{k}.class_w;
		updated_data_probs = compute_classifier_data_probs(data,class_w);
		expert_params{k}.data_probs = updated_data_probs;

		% just saving clust_v's for easy computation, needed below
		clust_v = expert_params{k}.clust_v;
		all_clust_v(k,:) = clust_v;

	end

		updated_data_wts = compute_classifier_data_wts(data,all_clust_v);		 

	for k = 1:K
		expert_params{k}.data_wts = updated_data_wts(k,:);
	end

end
plot(all_likelihood);
