function [p, h] = predictNN(X, varargin)
%PREDICTNN Predict the label of an input given a trained neural network
%
%   [p, h] = PREDICTNN(X, Theta, activationFn) outputs the predicted 
%   label of X given the trained weights of a neural network Theta. This
%   option does not normalize the data.
%   or 
%   [p, h] = PREDICTNN(X, modelNN)
%   This normalizes the data based on the mu and std in the modelNN
%
%   Vahe Tshitoyan
%   26/08/2017
%
%   Inputs
%   X:              The feature matrix to run the prediction on
%   Theta:          The weights as a cell array modelNN.Theta
%   activationFn:   'tanh' or 'sigm'
%   modelNN:        The trained NN model
%
%   Outputs:
%   p:              The predicted label
%   h:              The values of the hypotheses for all labels as a vector

if nargin==2
    modelNN = varargin{1};
    Theta = modelNN.Theta;
    activationFn = modelNN.activationFn;
    % also need to normalize here
    X = (X - repmat(modelNN.nnMu,size(X,1),1))...
        ./repmat(modelNN.nnSigma,size(X,1),1);
    X(isnan(X)) = 0;
elseif nargin==3
    Theta = varargin{1};
    activationFn = varargin{2};
else
    error('predictNN: Invalid Number of Arguments');
end

% choose the activation function
if strcmp(activationFn, 'tanh')
    aF = @tanh;
else % the default is sigmoid for now
    aF = @sigmoid;
end

% Number of examples
m = size(X, 1);

% hypothesis values of 1st layer
h = aF([ones(m, 1) X] * Theta{1}');
% hypothesis values of the consecutive layers
for ii=2:numel(Theta)
    h = aF([ones(m, 1) h] * Theta{ii}');
end

% prediction is the highest probability value
[~, p] = max(h, [], 2);
end
