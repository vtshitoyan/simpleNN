# simpleNN
An easy to use fully connected neural network library. Also see on [Matlab File Exchange](https://uk.mathworks.com/matlabcentral/fileexchange/64247-simple-neural-network).

## Example usages
### Basic

run the training

    modelNN = learnNN(X, y);

plot the confusion matrix for the validation set

    plotConfMat(modelNN.confusion_valid);
Here, `X` is an `[m x n]` feature matrix with `m` being the number of examples and `n` number of features. `y` is an `[m x 1]` vector of labels. `plotConfMat` plots the confusion matrix for the validation set.

### Custom

Set some custom options, including the layer structure, regularization parameter `lambda` and a choice of activation function.
    
    nnOptions = {'hiddenLayers', [40 20 10], 'lambda', 0.1, 'activationFn', 'tanh'};

Now, run the optimization using the custom options

    modelNN = learnNN(X, y, nnOptions);
