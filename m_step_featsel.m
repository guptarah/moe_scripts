function expert_params = m_step_featsel(expert_params,K,max_iter,iter_counter,eta_factor,data,targets_1inK)

% M step: COMPUTING THE CLASSIFIER PARAMETERS
for k = 1:K
        % computing the classifier weights w
        data_probs = expert_params{k}.data_probs;
        resp = expert_params{k}.resp;
        cur_w = expert_params{k}.class_w;
	cur_w_featsel = expert_params{k}.w_featsel;
	cur_v_featsel = expert_params{k}.v_featsel;

        eta1 = eta_factor^(floor(max_iter/(max_iter-iter_counter))); % set eta  
        [new_w] = update_w_featsel(data,data_probs,resp,cur_w,cur_w_featsel,targets_1inK,eta1);       
        expert_params{k}.class_w = new_w;               

        % computing the classifier weights v
        cur_v = expert_params{k}.clust_v;
        eta2 = eta_factor^(floor(max_iter/(max_iter-iter_counter)));            
        [new_v] = update_v_featsel(data,cur_v,cur_v_featsel,resp,eta2);
        expert_params{k}.clust_v = new_v;               

end

