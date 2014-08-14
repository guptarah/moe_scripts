function display_params(expert_params)

K = length(expert_params)

for k = 1:K
	plot(expert_params{k}.resp,'r');
	pause;
end
