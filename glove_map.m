% glove_mapping = load_glove('combined_embedding_0.mat'); 
% get_embeddings('hello how are you kuy98wrjhrgh hasugggsff8564430',300, glove_mapping)

function[glove] = glove_map(glove_mat_file)
    load(glove_mat_file)
    glove = wordMap;
end



