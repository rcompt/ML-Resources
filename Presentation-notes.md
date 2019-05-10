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
