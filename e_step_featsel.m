function [expert_params,all_likelihood] = e_step_featsel(expert_params,N,K,targets,iter_counter,all_likelihood)

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
all_likelihood(iter_counter) = sum(log(denom));

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
