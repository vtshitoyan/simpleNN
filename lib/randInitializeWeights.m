function w = randInitializeWeights(layers)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a neural network
%   given the structure of the layers. w is rolled back to a vector.
%
%   usage:  w = RANDINITIALIZEWEIGHTS(layers)
    
% number in each layer
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));
% function for determining the amplitude of init values for each layer
efun = @(x, y) sqrt(6)./(sqrt(x + y));
% the init apmlitudes for each layer
epsilon_init = repelem(bsxfun(efun, layers(1:end-1), layers(2:end)), nel);
% the init weights for each neuron
w = (2*rand(sum(nel), 1) - 1).*epsilon_init';
end
