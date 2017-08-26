function g = tanhGradient(z)
%TANHGRADIENT The gradient of tanh
%
%   usage:  g = TANHGRADIENT(z)
g = 1 - (tanh(z)).^2;
end

