function updated_data_probs = compute_classifier_data_probs_featsel(data,class_w,w_featsel)

updated_data_probs = exp((class_w .* w_featsel) * data');

normalizer = repmat(sum(updated_data_probs),size(updated_data_probs,1),1);

updated_data_probs = updated_data_probs ./normalizer;
