
# Modeling Procedure
## Data Splitting

### Even splitting across data sets
Need to ensure that data is not split such that each split contains varying distributions of classes and features
-Stratified sampling is good for class
Stratified sampling is the procedure in which a representative distribution of the target is present across all train, validation, and test data sets. 

### Stratified Splitting
In statistical surveys, when subpopulations within an overall population vary, it could be advantageous to sample each subpopulation (stratum) independently. Stratification is the process of dividing members of the population into homogeneous subgroups before sampling. The strata should define a partition of the population. That is, it should be collectively exhaustive and mutually exclusive: every element in the population must be assigned to one and only one stratum. Then simple random sampling or systematic sampling is applied within each stratum. The objective is to improve the precision of the sample by reducing sampling error. It can produce a weighted mean that has less variability than the arithmetic mean of a simple random sample of the population.

# Regularization
Regularization is the process of adding information in order to solve an ill-posed problem or to prevent overfitting. Regularization is a technique which makes slight modifications to the learning algorithm such that the model generalizes better. This in turn improves the model’s performance on the unseen data as well.

A theoretical justification for regularization is that it attempts to impose Occam's razor on the solution (as depicted in the figure above, where the green function, the simpler one, may be preferred). From a Bayesian point of view, many regularization techniques correspond to imposing certain prior distributions on model parameters.

L1 and L2 are the most common types of regularization. These update the general cost function by adding another term known as the regularization term.

Cost function = Loss (say, binary cross entropy) + Regularization term

Due to the addition of this regularization term, the values of weight matrices decrease because it assumes that a neural network with smaller weight matrices leads to simpler models. Therefore, it will also reduce overfitting to quite an extent.

However, this regularization term differs in L1 and L2.

## L1 

Cost Function = Loss + (Lambda/2m) * SUM(ABS(weights))

In this, we penalize the absolute value of the weights. Unlike L2, the weights may be reduced to zero here (acting like a feature selector). Hence, it is very useful when we are trying to compress our model. Otherwise, we usually prefer L2 over it.

## L2

Cost Function = Loss + (Lambda/2m) * SUM(weights**2)

Here, lambda is the regularization parameter. It is the hyperparameter whose value is optimized for better results. L2 regularization is also known as weight decay as it forces the weights to decay towards zero (but not exactly zero).

# Outliers

# Modeling Types

## Supervised v Unsupervised

## Generative v Discriminate

## Decision Trees
### Baseline Tree
### Ensemble Methods
#### XGBoosted Forests


## Linear Models

### OLS 

### Lasso and Ridge

### Logistic

## Neural Networks

#### Back Propagation

#### Feed Forward

#### Deep Learning

#### Convolutional NN

#### Recurrent NN

#### Generative Adversial Networks

#### ADAM

#### Dropout
This is the one of the most interesting types of regularization techniques. It also produces very good results and is consequently the most frequently used regularization technique in the field of deep learning.

So what does dropout do? At every iteration, it randomly selects some nodes and removes them along with all of their incoming and outgoing connections.

So each iteration has a different set of nodes and this results in a different set of outputs. It can also be thought of as an ensemble technique in machine learning.

Ensemble models usually perform better than a single model as they capture more randomness. Similarly, dropout also performs better than a normal neural network model.

This probability of choosing how many nodes should be dropped is the hyperparameter of the dropout function. As seen in the image above, dropout can be applied to both the hidden layers as well as the input layers.

#### Data Augmentation
The simplest way to reduce overfitting is to increase the size of the training data. In machine learning, we were not able to increase the size of training data as the labeled data was too costly.

But, now let’s consider we are dealing with images. In this case, there are a few ways of increasing the size of the training data – rotating the image, flipping, scaling, shifting, etc.

This technique is known as data augmentation. This usually provides a big leap in improving the accuracy of the model. It can be considered as a mandatory trick in order to improve our predictions.

#### Early Stopping
Early stopping is a kind of cross-validation strategy where we keep one part of the training set as the validation set. When we see that the performance on the validation set is getting worse, we immediately stop the training on the model. This is known as early stopping.

#### Vanishing Gradients

## Graphical Models

# Modeling Choices

# Issues
## Overfitting

## Underfitting


