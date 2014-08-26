function [expert_params,all_likelihood] = moe_train_featsel(data,targets,num_experts,max_iter,prob_simannl)

if nargin < 5
	prob_simannl = 0.01;
end

% data : each row contains one instance
% targets : discrete values 1 to M

if nargin < 4
	max_iter = 100;
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
	expert_params{k}.w_featsel = ones(1,D); 
	expert_params{k}.resp = (circshift(resp_pattern',k-1))';
	expert_params{k}.clust_v = ones(1,D);	
	expert_params{k}.v_featsel = ones(1,D);	
	expert_params{k}.data_probs = 1/M*ones(M,N); % a matrix where each row (corresponding to a class) contains data point probs 
	expert_params{k}.data_wts = ones(1,N); % contains the wt to multiply to above probs to find final prob
end


% learn the parameters
eta_factor = .2;
iter_counter = 0;
all_likelihood = zeros(max_iter-1,1);
prob_simannl1 = 0;
prob_simannl2 = 0;
while iter_counter < max_iter
	iter_counter = iter_counter + 1;

	if iter_counter > max_iter/3
		prob_simannl1 = prob_simannl;
		prob_simannl2 = prob_simannl;
	end

	% finding alternate w_featsel or v_feasel 
	if mod(iter_counter,2) == 0 % doing if for the individual experts
		aff_expert = randi(K);
		aff_expert_w_featsel = expert_params{aff_expert}.w_featsel;
		switch_vector = [rand(1,D-1) 1 ] < prob_simannl1; % last 1 for the bias term, should never be 0.			
		aff_expert_w_featsel_new = xor(aff_expert_w_featsel,switch_vector);		

		expert_params_old = expert_params;
		expert_params_new = expert_params;
		expert_params_new{aff_expert}.w_featsel = aff_expert_w_featsel_new;
	else % doing if for the feature space dividers
		aff_expert = randi(K);
		aff_expert_v_featsel = expert_params{aff_expert}.v_featsel;
		switch_vector = [rand(1,D-1) 1 ] < prob_simannl2; % last 1 for the bias term, should never be 0.
		aff_expert_v_featsel_new = xor(aff_expert_v_featsel,switch_vector);

		expert_params_old = expert_params;
		expert_params_new = expert_params;
		expert_params_new{aff_expert}.v_featsel = aff_expert_v_featsel_new;

	end

	for i = 1:2

		if i == 1
			expert_params_cur = expert_params_old;
		else
			expert_params_cur = expert_params_new;
		end

		% M step: COMPUTING THE CLASSIFIER PARAMETERS
		expert_params_cur = m_step_featsel(expert_params_cur,K,max_iter,iter_counter,eta_factor,data,targets_1inK);	
	
		% UPDATE DATA PROBS & DATA WTS
		all_clust_v = [];
		for k = 1:K
			class_w = expert_params_cur{k}.class_w;
			w_featsel = expert_params_cur{k}.w_featsel;
			updated_data_probs = compute_classifier_data_probs_featsel(data,class_w,w_featsel);
			expert_params_cur{k}.data_probs = updated_data_probs;
	
			% just saving clust_v's for easy computation, needed below
			clust_v = expert_params_cur{k}.clust_v;
			clust_v_featsel = expert_params_cur{k}.v_featsel;
			all_clust_v(k,:) = clust_v .* clust_v_featsel;
	
		end
	
			updated_data_wts = compute_classifier_data_wts(data,all_clust_v);		 
	
		for k = 1:K
			expert_params_cur{k}.data_wts = updated_data_wts(k,:);
		end
	
		% E step: COMPUTING RESPONSIBILITIES (should get the better simulated annealed feature set)
		if i == 1
	        	[expert_params_old, all_likelihood_old] = e_step_featsel(expert_params_cur,N,K,targets,iter_counter,all_likelihood);
		else
			[expert_params_new, all_likelihood_new] = e_step_featsel(expert_params_cur,N,K,targets,iter_counter,all_likelihood);
		end

	end

	if all_likelihood_old(iter_counter) > all_likelihood_new(iter_counter)
                expert_params = expert_params_old;
		all_likelihood = all_likelihood_old;
        else
                expert_params = expert_params_new;
		all_likelihood = all_likelihood_new;
        end

end
plot(all_likelihood);
