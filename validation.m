run matconvnet/matlab/vl_setupnn ;
validation_file = 'validation/validation_large.txt';
glove = glove_map('glove200d.mat');

load('parameters/best_parameters.mat', 'w_conv', 'b_conv','w_out', 'b_out', 'filter_size', 'n_filter', 'T', 'wordMap', 'd', 'is_glove');

[data]=read_validation_data(validation_file);
wordMap = update_wordmap(validation_file, wordMap);


T = updateT(T, wordMap, is_glove, glove, d);

validation_length = length(data);

data_validation = preprocess_data(data, wordMap, max(filter_size));

% validation accuracy
correct = 0;
t_vloss = 0;
for ind = 1:validation_length
    %get sentence matrix
    word_indexes = data_validation{ind,1};
    t = data_validation{ind, 2};
    X =T(word_indexes,:);

    [cache, pool_res, concat, o] = forward_prop(X, w_conv, b_conv, w_out, b_out, filter_size, n_filter);

    [~,predicted] = max(o);

    % compute loss
    v_loss = vl_nnloss(o, t);
    t_vloss = t_vloss + v_loss;

    if predicted == t
        correct = correct + 1;
    end
end

accuracy = correct/validation_length;
total_vloss = t_vloss/validation_length;
fprintf('validation accuracy: %f, validation loss: %f\n', accuracy, total_vloss);