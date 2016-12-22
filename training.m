run matconvnet/matlab/vl_setupnn ;
clear; clc;

%% Section 1: preparation before training

% section 1.1 read file 'train.txt', load data and vocabulary by using function read_data()
[data,wordMap]=read_data('train/train.txt');
wordMap('<PAD>') = length(wordMap) + 1;

% ------------------- Parameters to vary ------------------------
% init word embeding
d = 200;

% init filter
filter_size = [2,3,4,5];
n_filter = 200;

% learning parameters
learn_rate = 0.01; 
learn_rate_x = (learn_rate/n_filter)*length(filter_size)*10;
total_iteration = 40;

min_sentence_length = max(filter_size);

is_glove = 1;  
glove = glove_map('glove200d.mat');
%----------------------------------------------------------------

% section 1.2 seperate dataset to training and validation sets

train_length = length(data) * 0.80;
validation_length = length(data) - train_length;

data_indexes = preprocess_data(data,wordMap, min_sentence_length);

data_indexes = data_indexes(randperm(length(data_indexes)),:);
data_train = data_indexes(1:train_length,:);
data_validation = data_indexes(train_length+1:end,:);


%------------------- All Initialization ------------------------
T = prepareT(wordMap, true, glove, d);

w_conv = cell(length(filter_size), 1);
b_conv = cell(length(filter_size), 1);

for i=1:length(filter_size)
    w_conv{i} = normrnd(0, 0.1, [filter_size(i), d, 1, n_filter]);
    b_conv{i} = zeros(n_filter, 1);
end

% init output layer
total_filters = length(filter_size) * n_filter;
n_class = 2;
w_out = normrnd(0, 0.1, [total_filters, n_class]);
b_out = zeros(n_class, 1);
%-----------------------------------------------------------------

best_accuracy = 0;
for iter=1:total_iteration
    predicted = zeros(train_length,1);
    original = zeros(train_length,1); 
    t_loss = 0;
    t_vloss =0;
%% Section 2: training
% Note: 
% you may need the resouces [2-4] in the project description.
% you may need the follow MatConvNet functions: 
%       vl_nnconv(), vl_nnpool(), vl_nnrelu(), vl_nnconcat(), and vl_nnloss()
    
    % for each example in train.txt do  
    for ind = 1:train_length
         
        % section 2.1 forward propagation and compute the loss
        % Word embeddings
        word_index = data_train{ind,1};
        t = data_train{ind, 2};
        X = T(word_index,:);
        
        [cache, pool_res, concat, o] = forward_prop(X, w_conv, b_conv, w_out, b_out, filter_size, n_filter);
        
        original(i) = t;
        [~, predicted(i)] = max(o);
        
        % compute loss
        loss = vl_nnloss(o, t);
        t_loss = t_loss + loss; 
        
        % section 2.2 backward propagation and compute the derivatives
        dloss = vl_nnloss(o, t, 1);
        [dz, dw_out, db_out] = vl_nnconv(reshape(concat, [total_filters,1]), reshape(w_out,[total_filters,1,1,2]), b_out,dloss);       

        cache_conv = cell(3, length(filter_size));
        for i = 1: length(filter_size)
            dpool = dz(n_filter*(i-1)+1:i*n_filter);
            sizes = size(cache{1,i});
            drelu = vl_nnpool(cache{2,i}, [sizes(1), 1], reshape(dpool,[1,1,n_filter]));
            dconv = vl_nnrelu(cache{1,i}, drelu);
            [dx, dw_conv, db_conv] = vl_nnconv(X, w_conv{i}, b_conv{i}, dconv);
            
            % keep value for  parameters update
            cache_conv{1,i} = dx;
            cache_conv{2,i} = dw_conv;
            cache_conv{3,i} = db_conv;
        end
        
        % section 2.3 update the parameters
        % update parameters
        for i=1:length(filter_size)
            X = X - learn_rate_x*cache_conv{1,i};
            w_conv{i} = w_conv{i} - learn_rate*cache_conv{2,i};
            b_conv{i} = b_conv{i} - learn_rate*cache_conv{3,i};
        end
        for i=1:length(word_index)
            T(word_index(i),:) = X(i,:);
        end   
        
        % update output layer
        w_out = w_out - learn_rate*reshape(dw_out,[total_filters, n_class]);
        b_out = b_out - learn_rate*db_out;
    end
%% Section 3: evaluate prediction
    % train accuracy
    
    accuracy = length(find(predicted==original))/train_length;
    train_loss = t_loss/train_length;
    fprintf('%d,%f%%,%f\n', iter, accuracy*100, train_loss);
  
    % validation accuracy
    correct = 0;
    for ind = 1:validation_length
        %get sentence matrix
        word_index = data_validation{ind,1};
        t = data_validation{ind, 2};
        X =T(word_index,:);
        
        [cache, pool_res, concat, o] = forward_prop(X, w_conv, b_conv, w_out, b_out, filter_size, n_filter);
        
        [~,predict] = max(o);
        
        % compute loss
        v_loss = vl_nnloss(o, t);
        t_vloss = t_vloss + v_loss;
        
        if predict == t
            correct = correct + 1;
        end
    end
    
    accuracy = correct/validation_length;
    total_vloss = t_vloss/validation_length;
    fprintf('%f,%f\n', accuracy, total_vloss);
    
    if (accuracy > best_accuracy)
        best_accuracy = accuracy;
        fprintf('Storing best validation accuracy: %f \n', best_accuracy);
        save('parameters/best_parameters.mat', 'w_conv', 'b_conv','w_out', 'b_out', 'filter_size', 'n_filter', 'T', 'wordMap', 'd', 'is_glove')
    end
    
    
end