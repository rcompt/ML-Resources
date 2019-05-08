# Distributions

Probability distributions indicate the likelihood of an event or outcome. 

p(x) = the likelihood that random variable takes a specific value of x

The sum of all probabilities for all possible values must equal 1. Furthermore, the probability for a particular value or range of values must be between 0 and 1.

Probability distributions describe the dispersion of the values of a random variable. 

The kind of variable determines the type of probability distribution.

## Discrete 
Discrete probability functions are also known as probability mass functions and can assume a discrete number of values. 
Examples:
-Coin Tosses
-Counts of Events

### Uniform
Model multiple events with the same probability, such as rolling a die

### Binomial 
Model binary data, such as coin tosses or win-loss records

### Poisson
Model count data, such as the count of library book checkouts per hour

## Continuous
Continuous probability functions are also known as probability density functions. You know that you have a continuous distribution if the variable can assume an infinite number of values between any two values. Continuous variables are often measurements on a scale, such as height, weight, and temperature.

Unlike discrete probability distributions where each particular value has a non-zero likelihood, specific values in continuous distributions have a zero probability. For example, the likelihood of measuring a temperature that is exactly 32 degrees is zero. *** Double check this fact with other sources (May have to do with integrations as they return 0 is the bounds of the integral are equal... if I remember correctly).

## Goodness-of-Fit Tests

# Interview Questions

## Let A and B be events on the same sample space, with P (A) = 0.6 and P (B) = 0.7. Can these two events be disjoint?

These events cannot be disjoint because P(A) + P(B) > 1.

P(A || B) = P(A) + P(B) - P(A & B)

An event is disjoint if P(A & B) = 0. If A and B are disjoint P(A || B) = 0.6 + 0.7 = 1.3

And since probability cannot be greater than 1, these two mentioned events cannot be disjoint.

## Alice has 2 kids and one of them is a girl. What is the probability that the other child is also a girl? You can assume that there are an equal number of males and females in the world.

The outcomes for two kids can be {BB, BG, GB, GG}

Since it is mentioned that one of them is a girl, we can remove the BB option from the sample space. Therefore the sample space has 3 options while only one fits the second condition. Therefore the probability the second child will be a girl too is 1/3.

*** Need to check for a Bayesian solution to this

## A fair six-sided die is rolled twice. What is the probability of getting 2 on the first roll and not getting 4 on the second roll?
The two events mentioned are independent. The first roll of the die is independent of the second roll. Therefore the probabilities can be directly multiplied.

P(getting first 2) = 1/6

P(no second 4) = 5/6

Therefore P(getting first 2 and no second 4) = 1/6* 5/6 = 5/36



