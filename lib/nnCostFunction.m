function [J, grad] = nnCostFunction(nn_params, layers, X, y, lambda, activationFn)
%NNCOSTFUNCTION Implements the neural network cost function for a multi-layer
%   neural network which performs classification. It accepts 'sigm' or
%   'tanh' as activation functions. Computes the cost and gradient of the NN.
%   
%   usage: [J grad] = NNCOSTFUNCTION(nn_params, layers, X, y, lambda) 
%
%   Vahe Tshitoyan
%   26/08/2017
%
%   Input
%   nn_params:      current weights of the network, collected into a single
%                   vector
%   layers:         the structure of the NN, including the input layer and
%                   the output layer. E.g. [400 80 30 20 10]
%   X:              the feature matrix
%   y:              the labels (answers)
%   lambda:         regluarisation parameter
%   activationFn:   'tanh' or 'sigm'
%
%   Returns
%   J:              the value of the cost function
%   grad:           the gradients for each neuron

% unpacking into a cell array
Theta = unpackTheta(nn_params, layers);

% Numeber of examples
m = size(X, 1);
% number of layers excluding the output
nl = numel(layers) - 1;
% number of labels
num_labels = layers(end);
% number in each layer (seful for filling in gradients)
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));
snel = cumsum([0 nel]); % indices for boundaries

% Choose the activation function
if strcmp(activationFn, 'tanh')
    aF = @tanh;
    gF = @tanhGradient;
    % creating the matrix to hold vectorized y
    y_nn = -ones(m, num_labels);    
else % the default is sigmoid for now
    aF = @sigmoid;
    gF = @sigmoidGradient;
    % creating the matrix to hold vectorized y
    y_nn = zeros(m, num_labels);    
end

%% Forward Propogation
a_t = X; % the starting vals
a = cell(nl, 1);
z = cell(nl + 1, 1);
for ii=1:nl
    a{ii} = [ones(m, 1) a_t]; % adding the bias column
    z{ii+1} = a{ii}*Theta{ii}';
    a_t = aF(z{ii+1});
end
% hypotheses values
h_nn = a_t;

% Vectorized implementation
% converting y to the binary matrix of answers
idx = sub2ind([m num_labels],1:m,y');
y_nn(idx) = 1;

% computing the cost
if strcmp(activationFn, 'tanh')
    J_matrix = -((y_nn+1).*log((h_nn + 1)/2)/2 + (1 - (y_nn+1)/2).*log(1 - (h_nn+1)/2));
else % this one is sigmoid
    J_matrix = -(y_nn.*log(h_nn) + (1 - y_nn).*log(1 - h_nn));
end
J_matrix(isnan(J_matrix)) = 0;
J = sum(J_matrix(:));

% normalizing
J = J/m;

% adding the regularization term. The bias column is avoided
regularization_term = lambda * sum(cellfun(@(t) sum(sum(t(:,2:end).^2)), Theta)) / (2*m);
J = J + regularization_term;

%% Backpropogation
% preparing variables
delta = cell(nl + 1, 1);
Delta = cell(nl, 1);
ThetaGrad  =  cell(nl, 1);
grad = zeros(snel(end), 1);
% computing error terms for all training examples
% the last layer is special, so separate calc
delta{nl+1} = h_nn - y_nn;
Delta{nl} = delta{nl+1}' * a{nl}; % dimensions: s(bi) x (s(bi-1) + 1)
ThetaGrad{nl} = Delta{nl}/m + [zeros(layers(nl+1), 1) Theta{nl}(:, 2:end)]*lambda/m; % s(nl+1) x (s(nl) + 1)
grad((snel(nl)+1):snel(nl+1)) = ThetaGrad{nl}(:);
% the rest of the layers down to the input layer
for ii = 2:nl % starting from 2nd to last
    bi = nl - ii + 2; % backprop index
    delta{bi} = (delta{bi + 1} * Theta{bi}(:, 2:end)) .* gF(z{bi}); % dims: m x s(bi)
    Delta{bi-1} = delta{bi}' * a{bi-1}; % dimensions: s(bi) x (s(bi-1) + 1)
    ThetaGrad{bi-1} = Delta{bi-1}/m + [zeros(layers(bi), 1) Theta{bi-1}(:, 2:end)]*lambda/m; % s(bi) x (s(bi-1) + 1)
    grad((snel(bi - 1)+1):snel(bi)) = ThetaGrad{bi-1}(:);
end
end
