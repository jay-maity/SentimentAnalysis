 function [cache, pool_res, z, o] = forward_prop(X, w_conv, b_conv, w_out, b_out, filter_size, n_filter)
 % section 2.1 forward propagation and compute the loss
        %get sentence matrix
        
        total_filters = n_filter*length(filter_size);
        pool_res = cell(1, length(filter_size));
        cache = cell(2, length(filter_size));
        for i = 1: length(filter_size)
            %convolution operation
            conv = vl_nnconv(X, w_conv{i}, b_conv{i});
            
            %activation reLu
            relu = vl_nnrelu(conv);
            
            % 1-max pooling
            sizes = size(conv);
            pool = vl_nnpool(relu, [sizes(1), 1]);
            
            % keep values for back-propagate
            cache{2,i} = relu;
            cache{1,i} = conv;
            pool_res{i} = pool;
        end
        
        % concatenate
        z = vl_nnconcat(pool_res, 3);
        

        % compute output layer
        o = vl_nnconv(reshape(z, [total_filters,1]), reshape(w_out,[total_filters,1,1,2]), b_out);
 end