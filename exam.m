run matconvnet/matlab/vl_setupnn ;
clear; clc;

glove = glove_map('glove200d.mat');
is_header = 1;

test_file = 'test/test1.txt';
output_file = 'result/submission1.txt';
parameters_file = 'parameters/best_parameters.mat';

disp('Generating sample 1');
testing(test_file, output_file, glove, parameters_file, is_header);

test_file = 'test/test2.txt';
output_file = 'result/submission2.txt';
parameters_file = 'parameters/best_parameters.mat';

disp('Generating sample 2');
testing(test_file, output_file, glove, parameters_file, is_header);

test_file = 'test/test3.txt';
output_file = 'result/submission3.txt';
parameters_file = 'parameters/best_parameters.mat';

disp('Generating sample 3');
testing(test_file, output_file, glove, parameters_file, is_header);

disp('Thanks!');