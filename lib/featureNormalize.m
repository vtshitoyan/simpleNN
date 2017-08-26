function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1.
%
%   usage: [X_norm, mu, sigma] = FEATURENORMALIZE(X)

% column-wise mean
mu = mean(X);

% column-wise standard deviation
sigma = std(X);

% normalized matrix
X_norm = (X - repmat(mu,size(X, 1), 1))./repmat(sigma,size(X, 1), 1);
X_norm(isnan(X_norm)) = 0;

end
