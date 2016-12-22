function [T,wordMap] = prepareT(wordMap, is_glove, glove, d)

    if is_glove == 1
        % set word embeddings from GloVe
        keys = wordMap.keys();
        T = zeros(length(wordMap),d);
        for i = 1:length(keys)
            if isKey(glove,keys{i})
                T(wordMap(keys{i}),:) = glove(keys{i});
            else
                T(wordMap(keys{i}),:) = normrnd(0,0.1,[1,d]);
            end
        end
    else   
        % initialize word vector
        T = normrnd(0,0.1,[length(wordMap),d]);
    end
end