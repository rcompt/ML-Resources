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
The probability to obtain a similar or more extreme result than observed when the null hypothesis is assumed. ⇒ If the p-value is small, the null hypothesis is unlikely

A critical value is a point (or points) on the scale of the test statistic beyond which we reject the null hypothesis, and, is derived from the level of significance α of the test. Critical value can tell us, what is the probability of two sample means belonging to the same distribution. Higher, the critical value means lower the probability of two samples belonging to same distribution. The general critical value for a two-tailed test is 1.96, which is based on the fact that 95% of the area of a normal distribution is within 1.96 standard deviations of the mean. (Beware, this value is highly dependent on the same size, so 95% isn't a general threshold, it should be adjusted depending on the context). Critical values can be used to do hypothesis testing in following way

Calculate test statistic

Calculate critical values based on significance level alpha

Compare test statistic with critical values.


# Clustering
The free flow text data is first curated in the following stages:-

## Stage 1

Removing punctuations

Transforming to lower case

Grammatically tagging sentences and removing pre-identified stop phrases (Chunking)

Removing numbers from the document

Stripping any excess white spaces

## Stage 2
Removing generic words of the English language viz. determiners, articles, conjunctions and other parts of speech.

## Stage 3
Document Stemming which reduces each word to its root using Porter’s stemming algorithm.

These steps are best explained through the illustration below:-

Once all the documents in the corpus are transformed as explained above, a term document matrix is created and the documents are transformed into this vector space model using the 1-gram vectorizer (see below). Other more sophisticated implementations include n-gram (where n in a reasonably small integer)

## TF-IDF (Term Frequency – Inverse Document Frequency) Normalization
This is an optional step and can be performed in case there is high variability in the document corpus and the number of documents in the corpus is extremely large (of the order of several million). This normalization increases the importance of terms that appear multiple times in the same document while decreasing the importance of terms that appear in many documents (which would mostly be generic terms). The term weightages are computed as follows:-

## K-Means Clustering using Euclidean Distances
Post the TF-IDF transformation, the document vectors are put through a K-Means clustering algorithm which computes the Euclidean Distances amongst these documents and clusters nearby documents together.

## Auto-Tagging based on Cluster Centers
The algorithm then generates cluster tags, known as cluster centers which represent the documents contained in these clusters. The clustering and auto-generated tags are best depicted in the illustration below (Principal components 1 and 2 are plotted along the x and y axes respectively):-

In order for more and more users to benefit from this solution and analyze their unstructured text data, I have created a RESTful web service that users can access in two ways:

-A web interface for this service which is a Swagger API Docs front end. This is a very popular solution for RESTful web services. The user can navigate to the web interface URL, upload the data-set, specify the column containing the natural language data that needs to be analyzed and the desired number of clusters and within a few minutes the output will appear as a downloadable link containing the results of the analysis.

-Since the web service works on the concept of Application Programming Interface (API), the computation engine that performs the analysis is a separate component which is scalable, portable and can be accessed from any other application through RESTful HTTP.

Since all computations are performed in-memory, the results are lightning fast.



# Text Features
## N-Grams
An n-gram is a contiguous sequence of n items from a given sample of text or speech

## Skip-grams
The Skip-gram model architecture usually tries to achieve the reverse of what the CBOW model does. It tries to predict the source context words (surrounding words) given a target word (the center word). Considering our simple sentence from earlier, “the quick brown fox jumps over the lazy dog”. If we used the CBOW model, we get pairs of (context_window, target_word)where if we consider a context window of size 2, we have examples like ([quick, fox], brown), ([the, brown], quick), ([the, dog], lazy) and so on. Now considering that the skip-gram model’s aim is to predict the context from the target word, the model typically inverts the contexts and targets, and tries to predict each context word from its target word. Hence the task becomes to predict the context [quick, fox] given target word ‘brown’ or [the, brown] given target word ‘quick’ and so on. Thus the model tries to predict the context_window words based on the target_word.

Just like we discussed in the CBOW model, we need to model this Skip-gram architecture now as a deep learning classification model such that we take in the target word as our input and try to predict the context words.This becomes slightly complex since we have multiple words in our context. We simplify this further by breaking down each (target, context_words) pair into (target, context) pairs such that each context consists of only one word. Hence our dataset from earlier gets transformed into pairs like (brown, quick), (brown, fox), (quick, the), (quick, brown) and so on. But how to supervise or train the model to know what is contextual and what is not?

For this, we feed our skip-gram model pairs of (X, Y) where X is our input and Y is our label. We do this by using [(target, context), 1] pairs as positive input samples where target is our word of interest and context is a context word occurring near the target word and the positive label 1 indicates this is a contextually relevant pair. We also feed in [(target, random), 0] pairs as negative input samples where target is again our word of interest but random is just a randomly selected word from our vocabulary which has no context or association with our target word. Hence the negative label 0indicates this is a contextually irrelevant pair. We do this so that the model can then learn which pairs of words are contextually relevant and which are not and generate similar embeddings for semantically similar words.

### Implementing the Skip-gram Model
 
Let’s now try and implement this model from scratch to gain some perspective on how things work behind the scenes and also so that we can compare it with our implementation of the CBOW model. We will leverage our Bible corpus as usual which is contained in the norm_bible variable for training our model. The implementation will focus on five parts

Build the corpus vocabulary

Build a skip-gram [(target, context), relevancy] generator

Build the skip-gram model architecture

Train the Model

Get Word Embeddings


## [Word Embeddings](https://towardsdatascience.com/introduction-to-word-embedding-and-word2vec-652d0c2060fa)
Word Embedding => Collective term for models that learned to map a set of words or phrases in a vocabulary to vectors of numerical values.

General approach for dealing with words in your text data is to one-hot encode your text. You will have tens of thousands of unique words in your text vocabulary. Computations with such one-hot encoded vectors for these words will be very inefficient because most values in your one-hot vector will be 0. So, the matrix calculation that will happen in between a one-hot vector and a first hidden layer will result in a output that will have mostly 0 values

We use embeddings to solve this problem and greatly improve the efficiency of our network. Embeddings are just like a fully-connected layer. We will call this layer as— embedding layer and the weights as — embedding weights.

Now, instead of doing the matrix multiplication between the inputs and hidden layer we directly grab the values from embedding weight matrix. We can do this because the multiplication of one-hot vector with weight matrix returns the row of the matrix corresponding to the index of ‘1’ input unit

So, we use this Weight Matrix as lookup table. We encode the words as integers, for example ‘cool’ is encoded as 512, ‘hot’ is encoded as 764. Then to get hidden layer output value for ‘cool’ we just simply need to lookup the 512th row in the weight matrix. This process is called Embedding Lookup. The number of dimension in the hidden layer output is the embedding dimension

To reiterate :-

a) The embedding layer is just a hidden layer

b) The lookup table is just a embedding weight matrix

c) The lookup is just a shortcut for matrix multiplication

d) The lookup table is trained just like any weight matrix

Popular off-the-shelf word embedding models in use today:

Word2Vec (by Google)
GloVe (by Stanford)
fastText (by Facebook)

## Topic Models (clustering)
LDA is a probabilistic model capable of expressing uncertainty about the placement of topics across texts and the assignment of words to topics,

NMF is a deterministic algorithm which arrives at a single representation of the corpus. For this reason, NMF is often characterized as a machine learning algorithm.

Like LDA, NMF arrives at its representation of a corpus in terms of something resembling “latent topics”.

# Feature Engineering
The features in your data will directly influence the predictive models you use and the results you can achieve.

You can say that: the better the features that you prepare and choose, the better the results you will achieve. It is true, but it also misleading.

The results you achieve are a factor of the model you choose, the data you have available and the features you prepared. Even your framing of the problem and objective measures you’re using to estimate accuracy play a part. Your results are dependent on many inter-dependent properties.

You need great features that describe the structures inherent in your data.

Better features means flexibility.

You can choose “the wrong models” (less than optimal) and still get good results. Most models can pick up on good structure in data. The flexibility of good features will allow you to use less complex models that are faster to run, easier to understand and easier to maintain. This is very desirable.

Better features means simpler models.

With well engineered features, you can choose “the wrong parameters” (less than optimal) and still get good results, for much the same reasons. You do not need to work as hard to pick the right models and the most optimized parameters.

With good features, you are closer to the underlying problem and a representation of all the data you have available and could use to best characterize that underlying problem.

Better features means better results.

The algorithms we used are very standard for Kagglers. […]  We spent most of our efforts in feature engineering.

— Xavier Conort, on “Q&A with Xavier Conort” on winning the Flight Quest challenge on Kaggle

Feature engineering is the process of transforming raw data into features that better represent the underlying problem to the predictive models, resulting in improved model accuracy on unseen data.

You can see the dependencies in this definition:

The performance measures you’ve chosen (RMSE? AUC?)

The framing of the problem (classification? regression?)

The predictive models you’re using (SVM?)

The raw data you have selected and prepared (samples? formatting? cleaning?)

feature engineering is manually designing what the input x’s should be

# Feature Selection
Feature Selection is the process where you automatically or manually select those features which contribute most to your prediction variable or output in which you are interested in.

Having irrelevant features in your data can decrease the accuracy of the models and make your model learn based on irrelevant features.

How to select features and what are Benefits of performing feature selection before modeling your data?

· Reduces Overfitting: Less redundant data means less opportunity to make decisions based on noise.

· Improves Accuracy: Less misleading data means modeling accuracy improves.

· Reduces Training Time: fewer data points reduce algorithm complexity and algorithms train faster.

# Sampling (in relation to data splitting)
Stratification

# Cleaning
## Imputation
Median or Mean imputation, only good if data is missing at random

## Balancing
Over Under sampling

# Cross Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.

The procedure has a single parameter called k that refers to the number of groups that a given data sample is to be split into. As such, the procedure is often called k-fold cross-validation. When a specific value for k is chosen, it may be used in place of k in the reference to the model, such as k=10 becoming 10-fold cross-validation.

Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

It is a popular method because it is simple to understand and because it generally results in a less biased or less optimistic estimate of the model skill than other methods, such as a simple train/test split.

## The general procedure is as follows:

#### Shuffle the dataset randomly.
#### Split the dataset into k groups
#### For each unique group:
#### Take the group as a hold out or test data set
#### Take the remaining groups as a training data set
#### Fit a model on the training set and evaluate it on the test set
#### Retain the evaluation score and discard the model
#### Summarize the skill of the model using the sample of model evaluation scores

Importantly, each observation in the data sample is assigned to an individual group and stays in that group for the duration of the procedure. This means that each sample is given the opportunity to be used in the hold out set 1 time and used to train the model k-1 times.

This approach involves randomly dividing the set of observations into k groups, or folds, of approximately equal size. The first fold is treated as a validation set, and the method is fit on the remaining k − 1 folds.

— Page 181, An Introduction to Statistical Learning, 2013.

It is also important that any preparation of the data prior to fitting the model occur on the CV-assigned training dataset within the loop rather than on the broader data set. This also applies to any tuning of hyperparameters. A failure to perform these operations within the loop may result in data leakage and an optimistic estimate of the model skill.

Despite the best efforts of statistical methodologists, users frequently invalidate their results by inadvertently peeking at the test data.

— Page 708, Artificial Intelligence: A Modern Approach (3rd Edition), 2009.

The results of a k-fold cross-validation run are often summarized with the mean of the model skill scores. It is also good practice to include a measure of the variance of the skill scores, such as the standard deviation or standard error.

## Configuration of k
The k value must be chosen carefully for your data sample.

A poorly chosen value for k may result in a mis-representative idea of the skill of the model, such as a score with a high variance (that may change a lot based on the data used to fit the model), or a high bias, (such as an overestimate of the skill of the model).

Three common tactics for choosing a value for k are as follows:

Representative: The value for k is chosen such that each train/test group of data samples is large enough to be statistically representative of the broader dataset.
k=10: The value for k is fixed to 10, a value that has been found through experimentation to generally result in a model skill estimate with low bias a modest variance.
k=n: The value for k is fixed to n, where n is the size of the dataset to give each test sample an opportunity to be used in the hold out dataset. This approach is called leave-one-out cross-validation.
The choice of k is usually 5 or 10, but there is no formal rule. As k gets larger, the difference in size between the training set and the resampling subsets gets smaller. As this difference decreases, the bias of the technique becomes smaller

— Page 70, Applied Predictive Modeling, 2013.

A value of k=10 is very common in the field of applied machine learning, and is recommend if you are struggling to choose a value for your dataset.

To summarize, there is a bias-variance trade-off associated with the choice of k in k-fold cross-validation. Typically, given these considerations, one performs k-fold cross-validation using k = 5 or k = 10, as these values have been shown empirically to yield test error rate estimates that suffer neither from excessively high bias nor from very high variance.

— Page 184, An Introduction to Statistical Learning, 2013.

If a value for k is chosen that does not evenly split the data sample, then one group will contain a remainder of the examples. It is preferable to split the data sample into k groups with the same number of samples, such that the sample of model skill scores are all equivalent.

# Evaluation Metrics

No.	Evaluation Metric	Formula	Interpretation
1	Sensitivity	| A /(A + C)	| What percentage of all 1's were correctly predicted?
2	Specificity |	D/(B+D) |	What percentage of all 0's were correctly predicted?
3	Prevalence	| (A+C)/(A+B+C+D) |	Percentage of True 1's in the sample
4	Detection Rate	| A/(A+B+C+D) |	Correctly predicted 1's as a percentage of entire sample
5	Detection Prevalence	| (A+B)/(A+B+C+D) |	What percentage of the full sample was predicted as 1?
6	Balanced Accuracy	| (sensitivity+specificity)/2 |	A balance between correctly predicting the 1's and 0's
7	Precision	| A/(A+B) |	What percentage of predicted 1's are correct?
8	Recall	| A/(A+C) |	What percentage of all 1's were correctly predicted?
9	F1 Score	| 2 * Precision * Recall / (Precision + Recall) |	A combination of Precision and Recall
10	Cohen's Kappa	| (Observed Accuracy - Expected Accuracy) / (1 - Expected Accuracy) |	How the model exceeded random predictions in terms of accuracy
11	Concordance	| Proportion of Concordant Pairs |	Proportion of Concordant Pairs
12	Somers D	| (Concordant Pairs - Discordant Pairs - Ties) / Total Pairs |	A combination of concordance and discordance
13	AUROC	| Area Under the ROC Curve |	Model's true performance considering all possible probability cutoffs
14	Gini Coefficient |	(2 * AUROC) - 1 |	How the model exceeded random predictions in terms of ROC
15	KS Statistic |	Max(Cumulative% 1's - Cumulative% 0's) |	Used to decide how many customers to target
16	Youden's J Index	| Sensitivity + Specificity - 1 |	Similar to balanced accuracy

## Confusion Matrix
The rows in the confusion matrix are the count of predicted 0’s and 1’s (from y_pred), while, the columns are the actuals (from y_act).

So, you have 122 out of 133 benign instances predicted as benign and 70 out of 71 malignant instances predicted as malignant. This is good.

Secondly, look at the 1 in top-right of the table. This means the model predicted 1 instance as benign which was actually positive.

This is a classic case of ‘False Negative’ or Type II error. You want to avoid this at all costs, because, it says the patient is healthy when he is actually carrying malignant cells.

Also, the model predicted 11 instances as ‘Malignant’ when the patient was actually ‘Benign’. This is called ‘False Positive’ or Type I error. This condition should also be avoided but in this case is not as dangerous as Type II error.

## Sensitivity 
Sensitivity is the percentage of actual 1’s that were correctly predicted. It shows what percentage of 1’s were covered by the model.

## Specificity  
Specificity is the proportion of actual 0’s that were correctly predicted. So in this case, it is 122 / (122+11) = 91.73%.

Specificity matters more when classifying the 0’s correctly is more important than classifying the 1’s.

Maximizing specificity is more relevant in cases like spam detection, where you strictly don’t want genuine messages (0’s) to end up in spam (1’s).

## Detection rate
Detection rate is the proportion of the whole sample where the events were detected correctly. So, it is 70 / 204 = 34.31%.

## Precision
A high precision score gives more confidence to the model’s capability to classify 1’s.

## Recall 
Combining this with Recall gives an idea of how many of the total 1’s it was able to cover.

## F1-Score
A good model should have a good precision as well as a high recall. So ideally, I want to have a measure that combines both these aspects in one single metric – the F1 Score.

## Cohen's Kappa
Kappa is similar to Accuracy score, but it takes into account the accuracy that would have happened anyway through random predictions.

Kappa = (Observed Accuracy - Expected Accuracy) / (1 - Expected Accuracy)

Cohen's kappa is shown as an output of caret's confusionMatrix function.

## KS Statistic
The KS Statistic and the KS Chart (discussed next) are used to make decisions like: How many customers to target for a marketing campaign? or How many customers should we pay for to show ads etc.

So how to compute the Kolmogorov-Smirnov statistic?

Step 1: Once the prediction probability scores are obtained, the observations are sorted by decreasing order of probability scores. This way, you can expect the rows at the top to be classified as 1 while rows at the bottom to be 0's.

Step 2: All observations are then split into 10 equal sized buckets (bins).

Step 3: Then, KS statistic is the maximum difference between the cumulative percentage of responders or 1's (cumulative true positive rate) and cumulative percentage of non-responders or 0's (cumulative false positive rate).

The significance of KS statistic is, it helps to understand, what portion of the population should be targeted to get the highest response rate (1's).

The KS statistic can be computed using the ks_stat function in InformationValue package. By setting the returnKSTable = T, you can retrieve the table that contains the detailed decile level splits.

## ROC
Often, choosing the best model is sort of a balance between predicting the one's accurately or the zeroes accurately. In other words sensitivity and specificity.

But it would be great to have something that captures both these aspects in one single metric.

This is nicely captured by the 'Receiver Operating Characteristics' curve, also called as the ROC curve. In fact, the area under the ROC curve can be used as an evaluation metric to compare the efficacy of the models.

The area under the ROC curve is also shown. But how to interpret this plot?

Interpreting the ROC plot is very different from a regular line plot. Because, though there is an X and a Y-axis, you don't read it as: for an X value of 0.25, the Y value is .9.

Instead, what we have here is a line that traces the probability cutoff from 1 at the bottom-left to 0 in the top right.

This is a way of analyzing how the sensitivity and specificity perform for the full range of probability cutoffs, that is from 0 to 1.

Ideally, if you have a perfect model, all the events will have a probability score of 1 and all non-events will have a score of 0. For such a model, the area under the ROC will be a perfect 1.

So, if we trace the curve from bottom left, the value of probability cutoff decreases from 1 towards 0. If you have a good model, more of the real events should be predicted as events, resulting in high sensitivity and low FPR. In that case, the curve will rise steeply covering a large area before reaching the top-right.

Therefore, the larger the area under the ROC curve, the better is your model.

The ROC curve is the only metric that measures how well the model does for different values of prediction probability cutoffs. The optimalCutoff function from InformationValue can be used to know what cutoff gives the best sensitivity, specificity or both.

## Gini Coefficient
Gini Coefficient is an indicator of how well the model outperforms random predictions. It can be computed from the area under the ROC curve using the following formula:

Gini Coefficient = (2 * AUROC) - 1

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
