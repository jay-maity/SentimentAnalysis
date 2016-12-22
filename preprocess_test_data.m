function [data_indexes] = preprocess_test_data(data, wordMap, min_length)
    
    data_indexes = cell(length(data),2);
    for i=1:length(data)
        sentence = data{i,2};
        index = data{i,1};
        sentence_length = length(sentence);
        
        % pad sentence for size less than filter
        if sentence_length < min_length
            sentence_temp = cell(1, min_length);
            for j=1:sentence_length
                sentence_temp{j} = sentence{j};
            end
            for j=(sentence_length+1):min_length
                sentence_temp{j} = '<PAD>';
            end
            sentence = sentence_temp;
        end
        
        indexes = zeros(size(sentence));
        for j=1:length(indexes)
            indexes(j) = wordMap(sentence{j});

        end
        data_indexes{i,1} = indexes;
        data_indexes{i,2} = index;
    end
end