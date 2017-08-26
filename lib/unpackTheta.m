function [ Theta ] = unpackTheta( theta, layers )
%UNPACKTHETA given a vector theta and a vector of layer structure
%   unpacks theta into Theta{1}, Theta{2}, etc.. for each layer and 
%   retrurns a cell array Theta

% number of layers, excluding input
nl = numel(layers) - 1; 
% to store unpacked thetas
Theta = cell(nl, 1); 
% number in each layer
nel = bsxfun(@times, layers(1:end-1) + 1, layers(2:end));

% unpacking theta
for ii=1:nl
    Theta{ii} = reshape(theta(1:nel(ii)), layers(ii+1), layers(ii) + 1);
    theta(1:nel(ii)) = [];
end
end

