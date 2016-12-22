function[] = testing(test_file, output_file, glove, parameters_file, is_header)
    
    load(parameters_file, 'w_conv', 'b_conv','w_out', 'b_out', 'filter_size', 'n_filter', 'T', 'wordMap', 'd', 'is_glove');

    [data]=read_test_data(test_file);
    wordMap = update_wordmap(test_file, wordMap);

    T = updateT(T, wordMap, is_glove, glove, d);
    test_length = length(data);
    data_test = preprocess_test_data(data, wordMap, max(filter_size));


    fileID = fopen(output_file,'w');

    if is_header == 1
        fprintf(fileID,'id::label\n');
    end
    
    for ind = 1:test_length
        %get sentence matrix
        word_indices = data_test{ind,1};
        index = data_test{ind,2};

        X =T(word_indices,:);

        [cache, pool_res, concat, o] = forward_prop(X, w_conv, b_conv, w_out, b_out, filter_size, n_filter);

        [~,pred] = max(o);
        result = 0;
        if pred == 1
            result = 1;
        end 
        fprintf(fileID,'%d::%d\n', index, result);
    end
    fclose(fileID);
end