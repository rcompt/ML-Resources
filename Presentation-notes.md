# Pipelines
Pipeline is a powerful tool to standardise your operations and chain then in a sequence, make unions and finetune parameters.
Pipelines are a way to streamline a lot of the routine processes, encapsulating little pieces of logic into one function call, which makes it easier to actually do modeling instead just writing a bunch of code. Pipelines allow for experiments, and for a dataset like this that only has the text as a feature, you're going to need to do a lot of experiments. Plus, when your modeling gets really complicated, it's sometimes hard to see if you have any data leakage hiding somewhere. Pipelines are set up with the fit/transform/predict functionality, so you can fit a whole pipeline to the training data and transform to the test data, without having to do it individually for each thing you do. 

# Skewness handling
1. Precision & Recall
This isn’t really a solution to the problem, but it helps for evaluating the final model. As described in the fraud example, “accuracy” might not be the best metric for determining the quality of the model. Fortunately, the metrics “precision” and “recall” can help us out. Precision describes how many of the data records, which got classified as fraud, actually are illustrating fraudulent activities. On the other hand, recall refers to the percentage of correctly classified frauds based on the overall number of frauds of the data set.

2. Stratified Sampling
Once you split up the data into train, validation and test set, chances are close to 100% that your already skewed data becomes even more unbalanced for at least one of the three resulting sets. Think about it: Let’s say your data set contains 1000 records and of those 20 are labelled as “fraud”. As soon as you split up the data set into train, validation and test set, it is possible that the train set contains the majority of the “fraud”-records or maybe even all of them. Although not as likely, the same could happen for the validation or test set, which is even worse because then the machine learning algorithm has no chance at all to learn the hidden patterns of “fraud”-records. This is why you shouldn’t use random sampling when constructing the three different data sets, but stratified sampling instead. It assures that the train, validation and test sets are well balanced. Therewith the already existing problem of skewed classes is not intensified, which is what you want when creating high quality models. For R, the function “strata” of the package “sampling” provides functionality for stratified sampling.

3. Limit the Over-Represented Class
If you have a lot of data, then simply limit the allowed number of “not fraud” samples in your data set. Unfortunately in machine learning you can never have enough data and you usually have less than desperately needed. However, this method is also useful for smaller data sets as in most cases the left out data records wouldn’t add a substantial amount of information to the ones already included.

4. Weights/Cost
For most machine learning algorithms you can assign higher costs for “false positives” and “false negatives”. The higher the costs, the more the model is geared towards classifying something else. This way, you can influence either precision or recall and enhance the overall performance of the model. However, as one of the two metrics is going to become better, the other one inevitably will get worse. It’s a tradeoff. This principle of costs can also be applied to a multiclass classification scenario, where it is possible to assign different weights to specific classes. The higher the weight of a class, the more likely it is to be classified. For SVM in R, the argument “class.weights” of the “svm” function in the e1071 package allows for such an adjustment of the importance of each class. One more hint to ease understanding: To my knowledge weights are the same as reversed costs.

5. Treat it as an Anomaly Detection Problem
In anomaly detection, the basic idea is to predict the probability for every record to be an anomaly, e.g. fraud. This is done via looking at the values of the features of an unseen data record and comparing them to those of all other data records which are known to be normal. If the probability turns out to be below a certain threshold \boldsymbol{\epsilon}, for example 5%, then the data record is classified as an anomaly. Especially anomaly detection using a multivariate Gaussian distribution looks promising to me. However, anomaly detection cannot be applied to multiclass classification settings with skewed classes. The algorithm would only be able to tell which of the data records don’t belong to any of the labeled classes and therefore should be classified as something like “other”. Unfortunately, I don’t have any practical experience with anomaly detection algorithms yet and do neither know the performance nor the pitfalls of such an approach. But for further information, I can recommend the anomaly detection chapter of the Machine Learning class at coursera.

# P-values

# Clustering

# Text Features
## N-Grams
## Skip-grams
## Word Embeddings
## Topic Models (clustering)

# Feature Engineering

# Feature Selection

# Feature Filtering

# Sampling (in relation to data splitting)
Stratification

# Cleaning
## Imputation
## Balancing

# Cross Validation

# Evaluation Metrics

# Linear v Nonlinear

# Hyperparameter search

# Feature Importance

# Synthetic generation

# Outlier Analysis

# Models
## SVMs 
## RNNS
## XGBoost Forests
## Logistic Regression
### Odds ratios
