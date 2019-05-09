# Population v. Sample
In statistics “population” refers to the total set of observations that can be made. For eg, if we want to calculate average height of humans present on the earth, “population” will be the “total number of people actually present on the earth”.

A sample, on the other hand, is a set of data collected/selected from a pre-defined procedure. For our example above, it will be a small group of people selected randomly from some parts of the earth.

To draw inferences from a sample by validating a hypothesis it is necessary that the sample is random.

# Sampling Techniques

# When to choose a test


# Tests

## Null Hypothesis Testing
The null hypothesis is a general statement or default position that there is no relationship between two measured phenomena, or no association among groups.

The null hypothesis is generally assumed to be true until evidence indicates otherwise.

The concept of a null hypothesis is used differently in two approaches to statistical inference. In the significance testing approach of Ronald Fisher, a null hypothesis is rejected if the observed data are significantly unlikely to have occurred if the null hypothesis were true. In this case, the null hypothesis is rejected and an alternative hypothesis is accepted in its place.

### Critical Value
A critical value is a point (or points) on the scale of the test statistic beyond which we reject the null hypothesis, and, is derived from the level of significance α of the test. Critical value can tell us, what is the probability of two sample means belonging to the same distribution. Higher, the critical value means lower the probability of two samples belonging to same distribution. The general critical value for a two-tailed test is 1.96, which is based on the fact that 95% of the area of a normal distribution is within 1.96 standard deviations of the mean. (Beware, this value is highly dependent on the same size, so 95% isn't a general threshold, it should be adjusted depending on the context).
Critical values can be used to do hypothesis testing in following way

1. Calculate test statistic

2. Calculate critical values based on significance level alpha

3. Compare test statistic with critical values.

## Z-test
In a z-test, the sample is assumed to be normally distributed. A z-score is calculated with population parameters such as “population mean” and “population standard deviation” and is used to validate a hypothesis that the sample drawn belongs to the same population.

The statistics used for this hypothesis testing is called z-statistic, the score for which is calculated as

z = (x — μ) / (σ / √n), where

x= sample mean

μ = population mean

σ / √n = population standard deviation

## T-test

A t-test is used to compare the mean of two given samples. Like a z-test, a t-test also assumes a normal distribution of the sample. A t-test is used when the population parameters (mean and standard deviation) are not known.

There are three versions of t-test

1. Independent samples t-test which compares mean for two groups
2. Paired sample t-test which compares means from the same group at different times
3. One sample t-test which tests the mean of a single group against a known mean.
The statistic for this hypothesis testing is called t-statistic, the score for which is calculated as

t = (x1 — x2) / (σ / √n1 + σ / √n2), where

x1 = mean of sample 1

x2 = mean of sample 2

n1 = size of sample 1

n2 = size of sample 2

## ANOVA
ANOVA, also known as analysis of variance, is used to compare multiple (three or more) samples with a single test. There are 2 major flavors of ANOVA

1. One-way ANOVA: It is used to compare the difference between the three or more samples/groups of a single independent variable.

2. MANOVA: MANOVA allows us to test the effect of one or more independent variable on two or more dependent variables. In addition, MANOVA can also detect the difference in co-relation between dependent variables given the groups of independent variables.

The hypothesis being tested in ANOVA is

Null: All pairs of samples are same i.e. all sample means are equal

Alternate: At least one pair of samples is significantly different

The statistics used to measure the significance, in this case, is called F-statistics. The F value is calculated using the formula

F= ((SSE1 — SSE2)/m)/ SSE2/n-k, where

SSE = residual sum of squares

m = number of restrictions

k = number of independent variables

There are multiple tools available such as SPSS, R packages, Excel etc. to carry out ANOVA on a given sample.

## Chi-Square
Chi-square test is used to compare categorical variables. There are two type of chi-square test

1. Goodness of fit test, which determines if a sample matches the population.

2. A chi-square fit test for two independent variables is used to compare two variables in a contingency table to check if the data fits.

a. A small chi-square value means that data fits

b. A high chi-square value means that data doesn’t fit.

The hypothesis being tested for chi-square is

Null: Variable A and Variable B are independent

Alternate: Variable A and Variable B are not independent.

The statistic used to measure significance, in this case, is called chi-square statistic. The formula used for calculating the statistic is

Χ2 = Σ [ (Or,c — Er,c)2 / Er,c ] where

Or,c = observed frequency count at level r of Variable A and level c of Variable B

Er,c = expected frequency count at level r of Variable A and level c of Variable B


# P-values
The probability to obtain a similar or more extreme result than observed when the null hypothesis is assumed.
⇒ If the p-value is small, the null hypothesis is unlikely
# Time Series
