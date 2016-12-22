function [data_Index] = preprocess_data(data, wordMap, min_length)
    data_Index = cell(length(data),2);
    
    % get all sentences
    for i=1:length(data)
        sentence = data{i,2};
        sentence_length = length(sentence);
        
        % pad sentence shorter than filter length
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
        
        % collect all index for each word
        indexes = zeros(size(sentence));
        for j=1:length(indexes)
            indexes(j) = wordMap(sentence{j});
        end
        
        data_Index{i,1} = indexes;
        
        % label data 1 and 2 instead 0 and 1
        label = data{i,3};
        if label == 1
            label = 1;
        else
            label = 2;
        end
        data_Index{i,2} = label;
        
    end
end