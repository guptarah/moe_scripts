function updated_data_probs = compute_classifier_data_probs(data,class_w)

updated_data_probs = exp(class_w * data');

normalizer = repmat(sum(updated_data_probs),size(updated_data_probs,1),1);

updated_data_probs = updated_data_probs ./normalizer;
