1.[Modeling Procedure](#Modeling-Procedure)

2.[Regularization](#Regularization)

3.[Issues](#Issues)

4.(Modeling Types)[#Modeling-Types]

4a.(Clustering)[#Clustering]

# Modeling Procedure

## Cleaning 

### Setting up a Quality Plan
For any activity, a proper plan is very much necessary. Before you can go ahead with data cleaning, you need to define your expectations from the function. You need to define clear KPIs along with identifying areas where the data errors are more likely to occur and at the same time identifying the reasons for errors in the data. A solid plan will help you get started with your data cleaning process.

### Remove Unwanted observations
The first step to data cleaning is removing unwanted observations from your dataset.

This includes duplicate or irrelevant observations.

### Fill-out missing values
One of the first steps of fixing errors in your dataset is to find incomplete values and fill them out (Imputing). Most of the data that you may have can be categorized. In such cases, it is best to fill out your missing values based on different categories or create entirely new categories to include the missing values.

If your data are numerical, you can use mean and median to rectify the errors (depends if the missing data is at random for your choice). You can also take an average based on different criteria, — namely age, geographical location, etc., among others.

#### Missing categorical data
The best way to handle missing data for categorical features is to simply label them as ’Missing’!

You’re essentially adding a new class for the feature.
This tells the algorithm that the value was missing.
This also gets around the technical requirement for no missing values.

#### Missing numeric data
For missing numeric data, you should flag and fill the values.

Flag the observation with an indicator variable of missingness.
Then, fill the original missing value with 0 just to meet the technical requirement of no missing values.
By using this technique of flagging and filling, you are essentially allowing the algorithm to estimate the optimal constant for missingness, instead of just filling it in with the mean.

### Removing rows with missing values
One of the simplest things to do in data cleansing is to remove or delete rows with missing values. This may not be the ideal step in case of a huge amount of errors in your training data. If the missing values are considerably less, then removing or deleting missing values can be the right approach. You will have to be very sure that the data you are deleting does not include information that is present in the other rows of the training data.

### Fixing errors in the structure
Ensure there are no typographical errors and inconsistencies in the upper or lower case. Go through your data set, identify such errors, and solve them to make sure that your training set is completely error-free. This will help you to yield better results from your machine learning functions. Also, remove duplicate categorization from your data list and streamline your data.

### Filter Unwanted Outliers

Outliers can cause problems with certain types of models. For example, linear regression models are less robust to outliers than decision tree models.

In general, if you have a legitimate reason to remove an outlier, it will help your model’s performance.

However, outliers are innocent until proven guilty. You should never remove an outlier just because it’s a "big number." That big number could be very informative for your model.

We can’t stress this enough: you must have a good reason for removing an outlier, such as suspicious measurements that are unlikely to be real data.

### Reducing data for proper data handling
It is good to reduce the data you are handling. A downsized dataset can help you generate results that are more accurate. There are different ways of reducing data in your dataset. Whatever data records you have, sample them and choose the relevant subset from that data. This method of data handling is called Record Sampling. Apart from this method, you can also use Attribute Sampling. When it comes to the attribute sampling, select a subset of the most important attributes from the dataset.

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

## Clustering
There are many types of clustering algorithms, such as K means, fuzzy c- means, hierarchical clustering, etc. Other than these, several other methods have emerged which are used only for specific data sets or types (categorical, binary, numeric).

Among these different clustering algorithms, there exists clustering behaviors known as

Soft Clustering: In this technique, the probability or likelihood of an observation being partitioned into a cluster is calculated.
Hard Clustering: In hard clustering, an observation is partitioned into exactly one cluster (no probability is calculated).

### Distance Calculation for Clustering
There are some important things you should keep in mind:

With quantitative variables, distance calculations are highly influenced by variable units and magnitude. For example, clustering variable height (in feet) with salary (in rupees) having different units and distribution (skewed) will invariably return biased results. Hence, always make sure to standardize (mean = 0, sd = 1) the variables. Standardization results in unit-less variables.
Use of a particular distance measure depends on the variable types; i.e., formula for calculating distance between numerical variables is different than categorical variables.
Suppose, we are given a 2-dimensional data with xi = (xi1, xi2, . . . , xip) and xj = (xj1, xj2, . . . , xjp). Both are numeric variables. We can calculate various distances as follows:

1. Euclidean Distance: It is used to calculate the distance between quantitative (numeric) variables. As it involves square terms, it is also known as L2 distance (because it squares the difference in coordinates). Its formula is given by

                    d(xi , xj ) = (|xi1 − xj1|² + |xi2 − xj2|² + . . . + |xip − xjp|² ) 1/2
2. Manhattan Distance: It is calculated as the absolute value of the sum of differences in the given coordinates. This is known as L1 distance. It is also sometimes called the Minowski Distance.

An interesting fact about this distance is that it only calculates the horizontal and vertical distances. It doesn't calculate the diagonal distance. For example, in chess, we use the Manhattan distance to calculate the distance covered by rooks. Its formula is given by:

                    d(xi , xj ) = (|xi1 − xj1| + |xi2 − xj2| + . . . + |xip − xjp|
3. Hamming Distance: It is used to calculate the distance between categorical variables. It uses a contingency table to count the number of mismatches among the observations. If a categorical variable is binary (say, male or female), it encodes the variable as male = 0, female = 1.

In case a categorical variable has more than two levels, the Hamming distance is calculated based on dummy encoding. Its formula is given by (x,y are given points):

                      hdist(x, y) <- sum((x[1] != y[1]) + (x[2] != y[2]) + ...)
Here, a != b is defined to have a value of 1 if the expression is true, and a value of 0 if the expression is false. This is also known as the Jaccard Coefficient.

4. Gower Distance: It is used to calculate the distance between mixed (numeric, categorical) variables. It works this way: it computes the distance between observations weighted by its variable type, and then takes the mean across all variables.

Technically, the above-mentioned distance measures are a form of Gower distances; i.e. if all the variables are numeric in nature, Gower distance takes the form of Euclidean. If all the values are categorical, it takes the form of Manhattan or Jaccard distance. In R, ClusterOfVar package handles mixed data very well.

5. Cosine Similarity: It is the most commonly used similarity metric in text analysis. The closeness of text data is measured by the smallest angle between two vectors. The angle (Θ) is assumed to be between 0 and 90. A quick refresher: cos (Θ = 0) = 1 and cos (Θ = 90) = 0.

Therefore, the maximum dissimilarity between two vectors is measured at Cos 90 (perpendicular). And, two vectors are said to be most similar at Cos 0 (parallel). For two vectors (x,y), the cosine similarity is given by their normalized dot product shown below:

                     cossim(x, y) <- dot(x, y)/(sqrt(dot(x,x)*dot(y,y)))
Let's understand the clustering techniques now.



# Modeling Choices

# Issues
## Overfitting

## Underfitting

# Evaluation Metrics

