%% CMPT-741: sentiment analysis base on Convolutional Neural Network
% author: Jay Maity
% date: 15-11-2016

clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data, wordMap] = read_data();


% section 1.2 Initialization with random numbers
noof_wembed = 5;
T = init_training(wordMap, noof_wembed);

% section 1.3 Initialization of filters
filter_sizes = [2,3,4];
noof_filter_layer = 2;
[W_conv, B_conv, DZDW_conv, DZDW_conv_bias] = init_filters(filter_sizes, noof_filter_layer, noof_wembed);

% section 1.4 init output layer
total_filter = length(filter_sizes) * noof_filter_layer;
n_class = 2;
W_out = normrnd(0, 0.1, [total_filter, n_class]);
B_out = zeros(n_class, 1);

W_fc = normrnd(0, 0.1, [1,length(filter_sizes),noof_filter_layer,n_class]);
B_fc = zeros(n_class, 1);

% Delta sum
DZDW_output = zeros(1,length(filter_sizes));

%% Section 2: training
total_sentences = 6000;

% Setup MatConvNet.
run matlab/vl_setupnn ;

for sentence_no = 1: total_sentences
    
    % Get training set and result
    [X, result] = get_word_indexes(sentence_no, T, data, wordMap);
    
    % Setting the structure to cache backpropagation output of neurons
    conv_cache = cell(length(filter_sizes));
    relu_cache = cell(length(filter_sizes));
    pool_cache = cell(length(filter_sizes));
    
    for i = 1: length(filter_sizes)
        % convolution operation
        conv = vl_nnconv(X, W_conv{i}, B_conv{i});
        %importatnt: keeping the values for back propagation
        conv_cache{i} = conv;
        
        % apply activation function :relu
        relu = vl_nnrelu(conv);
        %importatnt: keeping the values for back propagation
        relu_cache{i} = relu;
        
        % 1-max pooling operation
        sizes = size(conv);
        pool = vl_nnpool(relu, [sizes(1), 1]);
        pool_cache{i} = pool;
    end

    concat = vl_nnconcat(pool_cache, 2);
    % use of vl_nnconv function to act as fully connected layer
    % https://github.com/vlfeat/matconvnet/issues/185
    conv_output = vl_nnconv(concat, filter_fully_connected, []);
    output_softmax = vl_nnsoftmax(conv_output);
    
    t_arr = data(sentence_no, 3);
    t = t_arr{1};
    
    if t == 0
        t= -1;
    end
       
    loss = vl_nnloss(output_softmax, t);
    
    % section 2.2 backward propagation and compute the derivatives
    dzdx_output = vl_nnloss(output_softmax, t, 1);
    [dzdx_outconv, dzdw_output] = vl_nnconv(concat, W_fc, [], dzdx_output);
    dzdx_concat = vl_nnconcat(pool_cache, 2, dzdx_outconv);

    dzdw_conv  = cell(length(filter_sizes), 1);

    for i = 1: length(filter_sizes)

        % 1-max pooling operation
        sizes = size(relu_cache{i});
        dzdx_pool = vl_nnpool(relu_cache{i}, [sizes(1), 1], dzdx_concat{i});

        % apply activation function :relu
        dzdx_relu = vl_nnrelu(conv_cache{i}, dzdx_pool);

        % convolution operation
        [dzdx_conv, dzdw_conv{i}] = vl_nnconv(X, W_conv{i}, B_conv{i}, dzdx_relu);
        
        % section 2.3 update the parameters
        DZDW_conv{i} = DZDW_conv{i} + dzdw_conv{i};
    end
    
    % section 2.3 update the parameters
    DZDW_output = DZDW_output + dzdw_output;
    
end


function [T] = init_training(wordMap, noof_wembed)
% Initialize training data with some initial value
%       wordMap(map), wordMap for all the words
% return: 
%       T(cell), matrix for all initial training input

    total_words = length(wordMap);

    % random sample from normal distribution
    % with mean = 0 , variance = 0.01
    T = normrnd(0, 0.1, [total_words, noof_wembed]);
end

function [W_conv, B_conv, DZDW_conv, DZDW_conv_bias] = init_filters(filter_sizes, noof_filter_layer, noof_wembed)
% Initialize training data with some initial value
%       filter_sizes(array), All the filter needs to be applied on input
%       layer
%       noof_filters, No of layers for each filter
% return: 
%       W_conv(cell), Initial weights to be applied on filters
%       B_conv(cell), Initial biases to be applied on filters

    W_conv = cell(length(filter_sizes), 1);
    B_conv = cell(length(filter_sizes), 1);
    DZDW_conv = cell(length(filter_sizes), 1);
    DZDW_conv_bias = cell(length(filter_sizes), 1);
    
    for i = 1: length(filter_sizes)
        % get filter size
        f = filter_sizes(i);
        % inint W with FW x FH x FC x K
        W_conv{i} = normrnd(0, 0.1, [f, noof_wembed, 1, noof_filter_layer]);
        B_conv{i} = zeros(noof_filter_layer, 1);
        DZDW_conv{i} = zeros(f, noof_wembed, 1, noof_filter_layer);
        DZDW_conv_bias = zeros(noof_filter_layer, 1);
    end
end

function [X, result] = get_word_indexes(sentence_no, T, data, wordMap)
% Get word indexes from master data
%       sentence_no(int), Sentence index (id)
%       data(cell), 1st column -> sentence id, 2nd column -> words, 3rd column -> label
%       T(cell), Initialized training set
%       wordMap(map), map to index for each number
% return: 
%       X(cell), matrix for training for one sentence
%       result(int), actual result
    
    % Get all words in the sentence
    word_cell = data(sentence_no, 2);
    word_texts = word_cell{1};
    for word_index = 1:length(word_texts)
        text = word_texts(word_index);
        word_indexes(word_index) = wordMap(text{1});
    end
    result = data(sentence_no, 3);
    X = T(word_indexes, :);
end