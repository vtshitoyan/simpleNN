function modelNN = learnNN(X, Y, varargin)
%LEARNNN Performs a neural network learning
%
%   Runs a fully connected neural network with backpropogation. One can
%   specify as many hidden layers as they like. See options for more
%   details on what can be done.
%
%   usage:              modelNN = LEARNNN(X, Y [, nnOptions])
%
%   Vahe Tshitoyan
%   26/08/2017
%
%   Input parameters
%   X:                  The feature matrix [m x f], where f is the number of
%                       features, and m is the number of training examples
%   Y:                  The vector of labels [m x 1]
%   nnOptions:          Options specified as pairs of values
%                       'lambda' - regularisaton parameter (numeric). The
%                       default is 0.
%                       'maxIter' - max number of iterations (numeric). The
%                       default is 500.
%                       'hiddenLayers' - row vector of the number of nodes
%                       in each hidden layer. E.g. [20 10 5] will create a
%                       network with 3 hidden layers of the given number of
%                       nodes (numeric). The default is a single layer with
%                       the number of nodes determined as a geometric 
%                       average of input and output layers 
%                       'nrOfLabels' - number of output labels (numeric). 
%                       The default is determined as the number of unique 
%                       values of y
%                       'activationFn' - either 'tanh' or 'sigm' (string).
%                       The default is 'sigm'
%                       'validPercent' - the percentage of examples randomly
%                       selected for validaton (numeric). The default is 20%.
%                       'doNormalize' - 0 or 1 (numeric). If true, the data
%                       will be normalized to 0 mean and 1 standard
%                       deviation.
%                       'nnMu' - mean of the data (numeric). Used only if
%                       doNormalize is set to false. The default is 0.
%                       'nnSigma' - standard deviation of the data 
%                       (numeric). Used only if doNormalize is set to false. 
%                       The default is 0.
%                       'savePath' - if set, the model will be saved at
%                       this location (string). The default is not set.
%
%   Output
%   modelNN:            The trained model. This includes confusion_train 
%                       and confusion_valid, the confusion matrices for
%                       training and validation sets. It also includes the
%                       resulting model and all the parameters that go with
%                       it.

% setting the default parameters
lambda = 0;
maxIter = 500;
nrOfLabels = numel(unique(Y));
hiddenLayers = round(sqrt(nrOfLabels*size(X, 2)));
activationFn = 'sigm';
validPercent = 20;
doNormalize = 1;
nnMu = 0;
nnSigma = 1;

% if options are supplied
if nargin>2
    % change the default options
    nnOptions = varargin{1};
    if ~mod(numel(nnOptions), 2)
        for ii=1:2:numel(nnOptions)
            if isnumeric(nnOptions{ii + 1})
                nOfn = numel(nnOptions{ii+1});
                if nOfn>1
                    eval(sprintf(['%s=[' repmat('%d '.', nOfn, 1)' '];'], nnOptions{ii}, nnOptions{ii+1}));
                else
                    eval(sprintf('%s=%d;', nnOptions{ii}, nnOptions{ii+1}));
                end
            else
                eval(sprintf('%s=''%s'';', nnOptions{ii}, nnOptions{ii+1}));
            end
            
        end
    else
        % thrown an error about number of arguments
        error('Number of Options should be Even.')
    end
end

input_layer_size  = size(X, 2);
layers = [input_layer_size, hiddenLayers, nrOfLabels]; % full layer structure;

modelNN.activationFn = activationFn;
modelNN.lambda = lambda;
modelNN.maxIter = maxIter;
modelNN.layers = layers;
modelNN.doNormalize = doNormalize;

% in case there are NaN elements
X(isnan(X))=0;
% making sure it starts from 1
Y = Y - min(Y) + 1; 

%% Splitting out the validation set
m = size(X, 1);
validation_set_size = round(m*validPercent/100); % e.g. 10% for the validation
rand_indices = randperm(m);
X = X(rand_indices, :);
Y = Y(rand_indices);
X_valid = X(1:validation_set_size, :);
Y_valid = Y(1:validation_set_size);
X(1:validation_set_size,:) = [];
Y(1:validation_set_size) = [];

% saving these with the model
modelNN.X = X;
modelNN.Y = Y;
modelNN.X_valid = X_valid;
modelNN.Y_valid = Y_valid;

%% Normalizing the features
if doNormalize
    X_norm = zeros(size(X));
    X_norm(:,1) = ones(size(X,1),1);
    [X_norm(:,1:end), nnMu, nnSigma] = featureNormalize(X(:,1:end));
else
    X_norm = X;
end

% initialize the weights
initial_nn_params = randInitializeWeights(layers);

% TODO: to be implemented in the future versions
% checkNNGradients(lambda, layers);

% Set options for fmincg
options = optimset('MaxIter', maxIter);

% Cost function for optimization
cf = @(p) nnCostFunction(p, layers, X_norm, Y, lambda, activationFn); 

% The actual minimization happens here
startT = cputime;
[theta, modelNN.costArray] = fmincg(cf, initial_nn_params, options);
elapsed = cputime - startT;
fprintf('Required CPU Time: %f\n', elapsed);

% unpacking the result to a cell array of weights for each layer
modelNN.Theta = unpackTheta( theta, layers );

%% Computing the Prediction Accuracy
% preparing the validaton set
if doNormalize
    X_valid_norm = (X_valid - repmat(nnMu,size(X_valid,1),1))...
                         ./repmat(nnSigma,size(X_valid,1),1);
else
    X_valid_norm = X_valid;
end
X_valid_norm(isnan(X_valid_norm))=0;

% Predictions on the validation and training sets
p_valid = predictNN(X_valid_norm, modelNN.Theta, modelNN.activationFn);
p_train = predictNN(X_norm, modelNN.Theta, modelNN.activationFn);

% Building confusion matrices
confusion_train = zeros(nrOfLabels, nrOfLabels);
confusion_valid = zeros(nrOfLabels, nrOfLabels);
% rows are the actual value
% columns are the predicted value
for lp=1:nrOfLabels
    for la=1:nrOfLabels
        confusion_valid(la, lp) = numel(find(p_valid==lp&Y_valid==la));
        confusion_train(la, lp) = numel(find(p_train==lp&Y==la));
    end
end

% filling in model params
modelNN.confusion_valid = confusion_valid;
modelNN.confusion_train = confusion_train;
modelNN.nnMu = nnMu;
modelNN.nnSigma = nnSigma;
modelNN.trainingTimestamp = datestr(now,'yy_mm_dd_HH_MM_SS');

if exist('savePath', 'var')
    % Saving the computed parameters
    save(savePath, 'modelNN');
end
end
