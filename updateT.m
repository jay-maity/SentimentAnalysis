function [T] = updateT(existingT, wordMap, is_glove, glove, d)

    if is_glove == 1
        % set word embeddings from GloVe
        keys = wordMap.keys();
        T = existingT;
        len_T = length(T);
        for i = len_T:length(keys)
            if isKey(glove,keys{i})
                T(i,:) = glove(keys{i});
            else
                T(i,:) = normrnd(0,0.1,[1,d]);
            end
        end
    else   
        % initialize word vector
        keys = wordMap.keys();
        T = existingT;
        len_T = length(T);
        for i = len_T:length(keys)
           T(wordMap(keys{i}),:) = normrnd(0,0.1,[1,d]);
        end
    end
end