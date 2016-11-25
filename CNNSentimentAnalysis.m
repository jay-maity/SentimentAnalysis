%% CMPT-741: sentiment analysis base on Convolutional Neural Network
% author: Jay Maity
% date: 24-11-2016

clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data] = load_training_data('train/train.txt');
glove_mapping = glove_map('combined_embedding_0.mat');

% section 1.2 Initialization with random numbers
noof_wembed = 300;
%T = init_training(wordMap, noof_wembed);

% section 1.3 Initialization of filters
filter_sizes = [2, 3, 4, 5];
noof_filter_layer = 200;
[W_conv, B_conv] = init_filters(filter_sizes, noof_filter_layer, noof_wembed);

% section 1.4 init output layer
n_class = 2;
[W_fc, B_fc] = init_fc_WB(filter_sizes, noof_filter_layer, n_class);

%% Section 2: training
training = [1000 6000];
start_train = training(1);
end_train = training(2);

% Setup MatConvNet.
run matconvnet/matlab/vl_setupnn ;

iterations = 1;
learning_rate = 0.1;

while(iterations > 0)

    for sentence_no = start_train: end_train

        row = data(sentence_no);
        index = data(sentence_no,1);
        index = index{1};
        sentence = data(sentence_no,2);
        sentence = sentence{1};
        result = data(sentence_no,3);
        result = result{1};
        
        % Get training set and result
        [X] = get_word_embeddings(sentence, noof_wembed, glove_mapping);

        % Setting the structure to cache backpropagation output of neurons
        conv_cache = cell(length(filter_sizes));
        relu_cache = cell(length(filter_sizes));
        pool_cache = cell(1, length(filter_sizes));

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

        concat = vl_nnconcat(pool_cache, 3);
        % use of vl_nnconv function to act as fully connected layer
        % https://github.com/vlfeat/matconvnet/issues/185
        conv_output = vl_nnconv(concat, W_fc, B_fc);
        output_softmax = vl_nnsoftmax(conv_output);

        t_arr = data(sentence_no, 3);
        t = t_arr{1};

        %loss = vl_nnloss(output_softmax, t);
        classfied = output_softmax;
        
        if (t == 1 && output_softmax(:,:,1) > output_softmax(:,:,2)) || (t == 0 && output_softmax(:,:,1) <= output_softmax(:,:,2))
            continue;
        end
         
        if t == 1
            classfied(:,:,1) = 1;
            classfied(:,:,2) = -1;
        else
            classfied(:,:,1) = -1;
            classfied(:,:,2) = 1;
        end
        
        % section 2.2 backward propagation and compute the derivatives
        [dzdx_outconv, dzdw_output, dzdw_output_bias] = vl_nnconv(concat, W_fc, B_fc, classfied);

        dzdx_concat = vl_nnconcat(pool_cache, 3, dzdx_outconv);

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
        end
        
        W_conv_old = W_conv;
        B_conv_old = B_conv;
        W_fc_old = W_fc;
        B_fc_old = B_fc;

        for i = 1: length(filter_sizes)      
            % inint W with FW x FH x FC x K
            W_conv{i} = W_conv{i} + (learning_rate * dzdw_conv{i});
            B_conv{i} = B_conv{i} + (learning_rate * dzdw_conv_bias{i});
        end

        W_fc = W_fc + (learning_rate * dzdw_output);
        B_fc = B_fc + (learning_rate * dzdw_output_bias);
        
        if isequal(W_conv, W_conv_old) && isequal(B_conv,B_conv_old) && isequal(W_fc, W_fc_old) && isequal(B_fc_old, B_fc)
            iterations = 0;
            break;
        end

    end
    
    disp(iterations)
    iterations = iterations - 1;
end

save('parameters/wparameters.mat','W_conv','B_conv', 'W_fc', 'B_fc', 'filter_sizes', 'noof_wembed');



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

function [W_fc, B_fc] = init_fc_WB(filter_sizes, noof_filter_layer, n_class)
% Initialize fully connected output layer
%       filter_sizes(array), All the filter needs to be applied on input
%       layer
%       noof_filters, No of layers for each filter
%       n_class, No of output classes
% return: 
%       W_fc(cell), Initial weights to be applied as convolution layer
%       B_fc(cell), Initial biases to be applied as convolution layer

    W_fc = normrnd(0, 0.1, [1,1, noof_filter_layer*length(filter_sizes), n_class]);
    B_fc = zeros(1, n_class);
end