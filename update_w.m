function [updated_w,del_cur_class_w] = update_w(data,data_probs,resp,cur_w,targets_1inK,eta1)

M = size(cur_w,1);
D = size(data,2);
N = size(data,1);

if nargin < 6
	eta1 = .01;
end

updated_w = zeros(size(cur_w));
for m = 1:M
	cur_class_w = cur_w(m,:);
	target_class_probs = targets_1inK(m,:);
	obt_class_probs = data_probs(m,:);
	diff_class_probs = obt_class_probs - target_class_probs;
	wtd_diff_class_probs = resp .* diff_class_probs;

	wting_mat = repmat( wtd_diff_class_probs,D,1)';
	wtd_data_points = wting_mat .* data;
	del_cur_class_w = sum(wtd_data_points);
	del_cur_class_w = del_cur_class_w/sum(abs(del_cur_class_w));


	updated_w(m,:) = cur_w(m,:) - (eta1*del_cur_class_w);	
	
end 
