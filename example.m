clear; close all; clc;

%% Setting up
% loading example data
testData = load('data/zero_to_ten_numbers.mat'); % numbers 0 to 9

% using the default options
nnOptions = {};

% % Alternative options
% nnOptions = {'lambda', 0.1,...
%             'maxIter', 50,...
%             'hiddenLayers', [40 20],...
%             'activationFn', 'tanh',...
%             'validPercent', 30,...
%             'doNormalize', 1};

%% Learning
modelNN = learnNN(testData.X, testData.y, nnOptions);
% plotting the confusion matrix for the validation set
plotConfMat(modelNN.confusion_valid);

%% Predicting on a random image
rI = randi(size(testData.X, 1)); % a random index
p = predictNN(testData.X(rI,:), modelNN); % the prediction

figure(1); cla(gca);
imagesc(reshape(testData.X(rI,:), 20, 20)); % plotting
title(sprintf('Actual: %d, Predicted: %d', ...
    mod(testData.y(rI), 10), mod(p, 10))); % index for number 0 is 10