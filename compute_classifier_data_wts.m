function updated_data_wts = compute_classifier_data_wts(data,all_clust_v);

updated_data_wts = exp(all_clust_v * data');

normalizer = repmat(sum(updated_data_wts),size(updated_data_wts,1),1);

updated_data_wts = updated_data_wts./normalizer;
