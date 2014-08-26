function updated_v = update_v_featsel(data,cur_v,cur_v_featsel,resp,eta2)

D = size(data,2);  
N = size(data,1);

if nargin < 4
	eta2 = .01;
end

term1 = data*cur_v';
term2 = log(resp)';

term1minus2 = term1 - term2;
wting_mat = repmat(term1minus2,1,D);
wtd_data_points = wting_mat .* data;

del_cur_v = cur_v_featsel .* sum(wtd_data_points);
del_cur_v = del_cur_v/sum(abs(del_cur_v));

updated_v = cur_v - (eta2*del_cur_v);

