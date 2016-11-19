X = [1 2 ;3 4;5 6;7 8];
W1_22 = normrnd(0, 0.1, [2, 2, 1, 2]);

W1_22(1,1,1,1) = 2;
W1_22(1,2,1,1) = 4;
W1_22(2,1,1,1) = 6;
W1_22(2,2,1,1) = 8;

W1_22(1,1,1,2) = 0.2;
W1_22(1,2,1,2) = 0.4;
W1_22(2,1,1,2) = 0.6;
W1_22(2,2,1,2) = 0.8;

W1_32 = normrnd(0, 0.1, [3, 2, 1, 2]);

W1_32(1,1,1,1) = 1;
W1_32(2,1,1,1) = 2;
W1_32(3,1,1,1) = 3;
W1_32(1,2,1,1) = 4;
W1_32(2,2,1,1) = 5;
W1_32(3,2,1,1) = 6;

W1_32(1,1,1,2) = 0.1;
W1_32(2,1,1,2) = 0.2;
W1_32(3,1,1,2) = 0.3;
W1_32(1,2,1,2) = 0.4;
W1_32(2,2,1,2) = 0.5;
W1_32(3,2,1,2) = 0.6;

B1 = zeros(1, 2);

conv_22 = vl_nnconv(X, W1_22, []);
conv_32 = vl_nnconv(X, W1_32, []);
relu_conv1 = vl_nnrelu(conv_22);
relu_conv2 = vl_nnrelu(conv_32);

% 1-max pooling operation
sizes = size(relu_conv1);
pool1 = vl_nnpool(relu_conv1, [sizes(1), 1]);

sizes = size(relu_conv2);
pool2 = vl_nnpool(relu_conv2, [sizes(1), 1]);

pool_res = cell(2);
pool_res{1} = pool1;
pool_res{2} = pool2;

concat_layer = vl_nnconcat(pool_res, 2);
W_filter_fully_connected = normrnd(0, 0.1, [1,2,2,2]);
% W_filter_fully_connected(:,:,1,1) = 1;
% W_filter_fully_connected(:,:,2,1) = 1;
% W_filter_fully_connected(:,:,1,2) = 1;
% W_filter_fully_connected(:,:,2,2) = 1;

output = vl_nnconv(concat_layer, W_filter_fully_connected, []);
disp(pool_res)