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
[W_conv, B_conv] = init_filters(filter_sizes, noof_filter_layer, noof_wembed);

% section 1.4 init output layer
n_class = 2;
[W_fc, B_fc] = init_fc_WB(filter_sizes, noof_filter_layer, n_class);

%% Section 2: training
training = [1 40];
total_sentences = training(2);

% Setup MatConvNet.
run matlab/vl_setupnn ;

iterations = 10;
learning_rate = 0.001;

while(iterations > 0)
    
    % Sum of gradient parameter initialization
    [DZDW_conv, DZDW_conv_bias] = init_filtersDXDW(filter_sizes, noof_filter_layer, noof_wembed);

    DZDW_output = zeros(1,length(filter_sizes));
    DZDW_output_bias = zeros(n_class, 1);

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
        conv_output = vl_nnconv(concat, W_fc, B_fc);
        output_softmax = vl_nnsoftmax(conv_output);

        t_arr = data(sentence_no, 3);
        t = t_arr{1};

%         if t == 0
%             t= -1;
%         end

        loss = vl_nnloss(output_softmax, t);
        classfied = output_softmax;
        
        if output_softmax(:,:,1) > output_softmax(:,:,2) && t == 1
            classfied(:,:,1) = 1;
            classfied(:,:,2) = 0;
        elseif output_softmax(:,:,1) <= output_softmax(:,:,2) && t == 0
            classfied(:,:,1) = 1;
            classfied(:,:,2) = 0;
        else
            classfied(:,:,1) = 0;
            classfied(:,:,2) = 1;
        end
        
        dxdz_softmax = vl_nnsoftmax(conv_output, classfied);
        
        % section 2.2 backward propagation and compute the derivatives
        dzdx_output = vl_nnloss(output_softmax, t, classfied);
        [dzdx_outconv, dzdw_output, dzdw_output_bias] = vl_nnconv(concat, W_fc, B_fc, dxdz_softmax);

        dzdx_concat = vl_nnconcat(pool_cache, 2, dzdx_outconv);

        dzdw_conv  = cell(length(filter_sizes), 1);
        dzdw_conv_bias  = cell(length(filter_sizes), 1);

        for i = 1: length(filter_sizes)

            % 1-max pooling operation
            sizes = size(relu_cache{i});
            dzdx_pool = vl_nnpool(relu_cache{i}, [sizes(1), 1], dzdx_concat{i});

            % apply activation function :relu
            dzdx_relu = vl_nnrelu(conv_cache{i}, dzdx_pool);

            % convolution operation
            [dzdx_conv, dzdw_conv{i}, dzdw_conv_bias{i}] = vl_nnconv(X, W_conv{i}, B_conv{i}, dzdx_relu);

            % section 2.3 update the parameters
            DZDW_conv{i} = DZDW_conv{i} + dzdw_conv{i};
            DZDW_conv_bias{i} = DZDW_conv_bias{i} + dzdw_conv_bias{i};
        end

        % section 2.3 update the parameters
        DZDW_output = DZDW_output + dzdw_output;
        DZDW_output_bias = DZDW_output_bias + dzdw_output_bias;

    end
    
    W_conv_old = W_conv;
    B_conv_old = B_conv;
    W_fc_old = W_fc;
    B_fc_old = B_fc;
    
    for i = 1: length(filter_sizes)      
        % inint W with FW x FH x FC x K
        W_conv{i} = W_conv{i} + (learning_rate * DZDW_conv{i});
        B_conv{i} = B_conv{i} + (learning_rate *DZDW_conv_bias{i});
    end
    
    W_fc = W_fc + (learning_rate * DZDW_output);
    B_fc = B_fc + (learning_rate * DZDW_output_bias);
    
    if isequal(W_conv, W_conv_old) && isequal(B_conv,B_conv_old) && isequal(W_fc, W_fc_old) && isequal(B_fc_old, B_fc)
        break;
    end
    
    iterations = iterations - 1;
end

validation = [41 47];
start_validation = validation(1);
end_validation = validation(2);

correct = 0;
for sentence_no = start_validation: end_validation
    % Get training set and result
    [X, result] = get_word_indexes(sentence_no, T, data, wordMap);
    predicted_result = predict(X, W_conv, B_conv, W_fc, B_fc, filter_sizes);
    
    if predicted_result == result{1}
        correct = correct + 1;
    end
end

disp(correct)


function result = predict(X, W_conv, B_conv, W_fc, B_fc, filter_sizes)

    pool_cache = cell(length(filter_sizes));
    for i = 1: length(filter_sizes)
        % convolution operation
        conv = vl_nnconv(X, W_conv{i}, B_conv{i});
        
        % apply activation function :relu
        relu = vl_nnrelu(conv);
        
        % 1-max pooling operation
        sizes = size(conv);
        pool_cache{i} = vl_nnpool(relu, [sizes(1), 1]);
     end

    concat = vl_nnconcat(pool_cache, 2);
    % use of vl_nnconv function to act as fully connected layer
    % https://github.com/vlfeat/matconvnet/issues/185
    conv_output = vl_nnconv(concat, W_fc, B_fc);
    output_softmax = vl_nnsoftmax(conv_output);
    
    if output_softmax(:,:,1) > output_softmax(:,:,2)
        result = 1;
    else
        result = 0;
    end
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

function [W_conv, B_conv] = init_filters(filter_sizes, noof_filter_layer, noof_wembed)
% Initialize training data with some initial value
%       filter_sizes(array), All the filter needs to be applied on input
%       layer
%       noof_filters, No of layers for each filter
% return: 
%       W_conv(cell), Initial weights to be applied on filters
%       B_conv(cell), Initial biases to be applied on filters

    W_conv = cell(length(filter_sizes), 1);
    B_conv = cell(length(filter_sizes), 1);

    for i = 1: length(filter_sizes)
        % get filter size
        f = filter_sizes(i);
        % inint W with FW x FH x FC x K
        W_conv{i} = normrnd(0, 0.1, [f, noof_wembed, 1, noof_filter_layer]);
        B_conv{i} = zeros(noof_filter_layer, 1);
    end
end

function [DZDW_conv, DZDW_conv_bias] = init_filtersDXDW(filter_sizes, noof_filter_layer, noof_wembed)
% Initialize gradent sum
%       filter_sizes(array), All the filter needs to be applied on input
%       layer
%       noof_filters, No of layers for each filter
% return: 
%       DZDW_conv(cell), Initial weights to be applied on filters
%       DZDW_conv_bias(cell), Initial biases to be applied on filters

    DZDW_conv = cell(length(filter_sizes), 1);
    DZDW_conv_bias = cell(length(filter_sizes), 1);
    
    for i = 1: length(filter_sizes)
        % get filter size
        f = filter_sizes(i);
        
        % inint W with FW x FH x FC x K
        DZDW_conv{i} = zeros(f, noof_wembed, 1, noof_filter_layer);
        DZDW_conv_bias{i} = zeros(noof_filter_layer, 1);
    end
end

function [W_fc, B_fc] = init_fc_WB(filter_sizes, noof_filter_layer, n_class)
% Initialize fully connected output layer
%       filter_sizes(array), All the filter needs to be applied on input
%       layer
%       noof_filters, No of layers for each filter
%       n_class, No of output classes
% return: 
%       W_fc(cell), Initial weights to be applied as convolution layer
%       B_fc(cell), Initial biases to be applied as convolution layer

    W_fc = normrnd(0, 0.1, [1,length(filter_sizes), noof_filter_layer, n_class]);
    B_fc = zeros(n_class, 1);
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
    word_indexes = [];
    word_cell = data(sentence_no, 2);
    word_texts = word_cell{1};
    for word_index = 1:length(word_texts)
        text = word_texts(word_index);
        word_indexes(word_index) = wordMap(text{1});
    end
    result = data(sentence_no, 3);
    X = T(word_indexes, :);
end