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

### K means Clustering
This technique partitions the data set into unique homogeneous clusters whose observations are similar to each other but different than other clusters. The resultant clusters remain mutually exclusive, i.e., non-overlapping clusters.

In this technique, "K" refers to the number of cluster among which we wish to partition the data. Every cluster has a centroid. The name "k means" is derived from the fact that cluster centroids are computed as the mean distance of observations assigned to each cluster.

This k value k is given by the user. It's a hard clustering technique, i.e., one observation gets classified into exactly one cluster.

Technically, the k means technique tries to classify the observations into K clusters such that the total within cluster variation is as small as possible. Let C denote a cluster. k denotes the number of clusters. So, we try to minimize:

                      minimum(Cι..Cκ){∑W(Cκ)}
But, how do we calculate within cluster variation? The most commonly used method is squared Euclidean distance. In simple words, it is the sum of squared Euclidean distance between observations in a cluster divided by the number of observations in a cluster (shown below): within cluster variation formula

|Cκ| denotes the number of observations in a cluster.

Now, we fairly understand what goes behind k means clustering. But how does it start ? Practically, it takes the following steps:

1.Let's say the value of k = 2. At first, the clustering technique will assign two centroids randomly in the set of observations.
2.Then, it will start partitioning the observations based on their distance from the centroids. Observations relatively closer to any centroid will get partitioned accordingly.
3.Then, based on the number of iterations (after how many times we want the algorithm to converge) we've given, the cluster centroids will get recentered in every iteration. With this, the algorithm will try to continually optimize for the lowest within cluster variation.
4.Based on the newly assigned centroids, assign each observation falling closest to the new centroids.
5.This process continues until the cluster centers do not change or the stopping criterion is reached.

#### How to select best value of k in k means?
Determining the best value of k plays a critical role in k means model performance. To select best value of k, there is no "one rule applies all situation." It depends on the shape and distribution of the observations in a data set.

Intuitively, by finding the best value of k, we try to find a balance between the number of clusters and the average variation within a cluster.

Following are the methods useful to find an optimal k value:

1. Cross Validation: It's a commonly used method for determining k value. It divides the data into X parts. Then, it trains the model on X-1 parts and validates (test) the model on the remaining part.

The model is validated by checking the value of the sum of squared distance to the centroid. This final value is calculated by averaging over X clusters. Practically, for different values of k, we perform cross validation and then choose the value which returns the lowest error.

2. Elbow Method: This method calculates the best k value by considering the percentage of variance explained by each cluster. It results in a plot similar to PCA's scree plot. In fact, the logic behind selecting the best cluster value is the same as PCA.

In PCA, we select the number of components such that they explain the maximum variance in the data. Similarly, in the plot generated by the elbow method, we select the value of k such that percentage of variance explained is maximum.

3. Silhouette Method: It returns a value between -1 and 1 based on the similarity of an observation with its own cluster. Similarly, the observation is also compared with other clusters to derive at the similarity score. High value indicates high match, and vice versa. We can use any distance metric (explained above) to calculate the silhouette score.

4. X means Clustering: This method is a modification of the k means technique. In simple words, it starts from k = 1 and continues to divide the set of observations into clusters until the best split is found or the stopping criterion is reached. But, how does it find the best split ? It uses the Bayesian information criterion to decide the best split.

### Hierarchical Clustering 
# Modeling Choices
In simple words, hierarchical clustering tries to create a sequence of nested clusters to explore deeper insights from the data. For example, this technique is being popularly used to explore the standard plant taxonomy which would classify plants by family, genus, species, and so on.

Hierarchical clustering technique is of two types:

1. Agglomerative Clustering – It starts with treating every observation as a cluster. Then, it merges the most similar observations into a new cluster. This process continues until all the observations are merged into one cluster. It uses a bottoms-up approach (think of an inverted tree).

2. Divisive Clustering – In this technique, initially all the observations are partitioned into one cluster (irrespective of their similarities). Then, the cluster splits into two sub-clusters carrying similar observations. These sub-clusters are intrinsically homogeneous. Then, we continue to split the clusters until the leaf cluster contains exactly one observation. It uses a top-down approach.

Here we’ll study about agglomerative clustering (it's the most commonly used).

This technique creates a hierarchy (in a recursive fashion) to partition the data set into clusters. This partitioning is done in a bottoms-up fashion. This hierarchy of clusters is graphically presented using a Dendogram.

This dendrogram shows how clusters are merged / split hierarchically. Each node in the tree is a cluster. And, each leaf of the tree is a singleton cluster (cluster with one observation). So, how do we find out the optimal number of clusters from a dendrogram?

Let’s understand how to study a dendrogram.

As you know, every leaf in the dendrogram carries one observation. As we move up the leaves, the leaf observations begin to merge into nodes (carrying observations which are similar to each other). As we move further up, these nodes again merge further.

Always remember, lower the merging happens (towards the bottom of the tree), more similar the observations will be. Higher the merging happens (toward the top of the tree), less similar the observations will be.

To determine clusters, we make horizontal cuts across the branches of the dendrogram. The number of clusters is then calculated by the number of vertical lines on the dendrogram, which lies under horizontal line.

the horizontal line cuts the dendrogram into three clusters since it surpasses three vertical lines. In a way, the selection of height to make a horizontal cut is similar to finding k in k means since it also controls the number of clusters.

But, how to decide where to cut a dendrogram? Practically, analysts do it based on their judgement and business need. More logically, there are several methods (described below) using which you can calculate the accuracy of your model based on different cuts. Finally, select the cut with a better accuracy.

The advantage of using hierarchical clustering over k means is, it doesn't require advanced knowledge of number of clusters. However, some of the advantages which k means has over hierarchical clustering are as follows:

-It uses less memory.
-It converges faster.
-Unlike hierarchical, k means doesn't get trapped in mistakes made on a previous level. It improves iteratively.
-k means is non-deterministic in nature, i.e.. after every time you initialize, it will produce different clusters. On the contrary, hierarchical clustering is deterministic.
Note: K means is preferred when the data is numeric. Hierarchical clustering is preferred when the data is categorical.

#### What are the evaluation methods used in cluster analysis ?
In supervised learning, we are given a target variable to calculate the model's accuracy. But, what do you do when there is no target variable? How can we benchmark the model's performance? The methods to evaluate clustering accuracy can be divided into two broad categories:

*Internal Accuracy Measures: *As the name suggests, these measures calculate the cluster's accuracy based on the compactness of a cluster. Following are the methods which fall under this category:

1. Sum of Squared Errors (SSE) - The compactness of a cluster can be determined by calculating its SSE. It works best when the clusters are well separated from one another. Its formula is given by (shown below) where Cκ is the number of observations in a cluster. µκ is the mean distance in cluster k.  sum of squared errors formula for clustering

2. Scatter Criteria - In simple words, it calculates the spread of a cluster. To do that, first it calculates a scatter matrix, within cluster scatter and between cluster scatter. Then, it sums over the resulting values to derive total scatter values. Lower values are desirable. Let's understand their respective formulas:

The scatter matrix is the  sum of [product of (difference between the observation and its cluster mean) and (its transpose)]. It is calculated by

clustering evaluation technique

Within cluster scatter (Sω) is simply the sum of all Sκ values. The between cluster matrix (SB) can be calculated as

clustering formula R

where Nκ is the number of observations in the k cluster and µ is the total mean vector calculated as

cluster mean formula

where m is the number of matches in the cluster. Finally, the total scatter metric S(T) can be calculated as

                        S(T) = Sω + SΒ
*External Accuracy Measures: *These measures are calculated by matching the structure of the clusters with some pre-defined classification of instances in the data. Let's look at these measures:

1. Rand Index - It compares the two clusters and tries to find the ratio of matching and unmatched observations among two clustering structures (C1 and C2). Its value lies between 0 and 1. Think of the clustering structures (C1 and C2) with several small clusters. Think of C1 as your predicted cluster output and C2 as the actual cluster output. Higher the value, better the score. Its simple formula is given by:

RAND SCORE = a + d / (a + b + c + d)

where

a = observations which are available in the same cluster in both structures (C1 and C2)
b = observations which are available in a cluster in C1 and not in the same cluster in C2
c = observations which are available in a cluster in C2 and not in the same cluster in C1
d = observations which are available in different clusters in C1 and C2
2. Precision Recall Measure -  This metric is derived from the confusion matrix. Recall is also known as Sensitivity [True Positive/ (True Positive + False Negative)]. For clustering, we use this measure from an information retrieval point of view. Here, precision is a measure of correctly retrieved items. Recall is measure of matching items from all the correctly retrieved items.


# Issues
Understanding model fit is important for understanding the root cause for poor model accuracy. This understanding will guide you to take corrective steps. We can determine whether a predictive model is underfitting or overfitting the training data by looking at the prediction error on the training data and the evaluation data.

-Overfitting: too much reliance on the training data

-Underfitting: a failure to learn the relationships in the training data

-High Variance: model changes significantly based on training data

-High Bias: assumptions about model lead to ignoring training data

-Overfitting and underfitting cause poor generalization on the test set

-A validation set for model tuning can prevent under and overfitting

## Overfitting
Your model is overfitting your training data when you see that the model performs well on the training data but does not perform well on the evaluation data. This is because the model is memorizing the data it has seen and is unable to generalize to unseen examples.
 
## Underfitting
Your model is underfitting the training data when the model performs poorly on the training data. This is because the model is unable to capture the relationship between the input examples (often called X) and the target values (often called Y). 

# Evaluation Metrics

## Type 1 Error
False Positive

## Type 2 Error
False Negative

## Sensitivity, Recall, Hit rate, or True Positive Rate (TPR)
Sensitivity is the percentage of actual 1’s that were correctly predicted. It shows what percentage of 1’s were covered by the model.

The total number of 1’s is 71 out of which 70 was correctly predicted. So, sensitivity is 70/71 = 98.59%

Sensitivity matters more when classifying the 1’s correctly is more important than classifying the 0’s. Just like what we need here in BreastCancer case, where you don’t want to miss out any malignant to be classified as ‘benign’.

## Specificity, Selectivity or True Negative Rate (TNR)
Specificity is the proportion of actual 0’s that were correctly predicted. So in this case, it is 122 / (122+11) = 91.73%.

Specificity matters more when classifying the 0’s correctly is more important than classifying the 1’s.

Maximizing specificity is more relevant in cases like spam detection, where you strictly don’t want genuine messages (0’s) to end up in spam (1’s).

Detection rate is the proportion of the whole sample where the events were detected correctly. So, it is 70 / 204 = 34.31%.

## Precision or Positive Predictive Value (PPV)
