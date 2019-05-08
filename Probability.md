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

## Bobo the amoeba has a 25%, 25%, and 50% chance of producing 0, 1, or 2 o spring, respectively. Each of Bobo’s descendants also have the same probabilities. What is the probability that Bobo’s lineage dies out?
  - p=1/4+1/4*p+1/2*p^2 => p=1/2
## In any 15-minute interval, there is a 20% probability that you will see at least one shooting star. What is the proba- bility that you see at least one shooting star in the period of an hour?
  - 1-(0.8)^4. Or, we can use Poisson processes
## How can you generate a random number between 1 - 7 with only a die?
* Launch it 3 times: each throw sets the nth bit of the result. 
* For each launch, if the value is 1-3, record a 0, else 1.
The result is between 0 (000) and 7 (111), evenly spread (3 independent throw). Repeat the throws if 0 was obtained: the process stops on evenly spread values.
## How can you get a fair coin toss if someone hands you a coin that is weighted to come up heads more often than tails?
  - Flip twice and if HT then H, TH then T.
## You have an 50-50 mixture of two normal distributions with the same standard deviation. How far apart do the means need to be in order for this distribution to be bimodal?
  - more than two standard deviations
## Given draws from a normal distribution with known parameters, how can you simulate draws from a uniform distribution?
  - plug in the value to the CDF of the same random variable
## A certain couple tells you that they have two children, at least one of which is a girl. What is the probability that they have two girls?
  - 1/3
## You have a group of couples that decide to have children until they have their first girl, after which they stop having children. What is the expected gender ratio of the children that are born? What is the expected number of children each couple will have?
  - gender ratio is 1:1. Expected number of children is 2. let X be the number of children until getting a female (happens with prob 1/2). this follows a geometric distribution with probability 1/2
## How many ways can you split 12 people into 3 teams of 4?
  - the outcome follows a multinomial distribution with n=12 and k=3. but the classes are indistinguishable
## Your hash function assigns each object to a number between 1:10, each with equal probability. With 10 objects, what is the probability of a hash collision? What is the expected number of hash collisions? What is the expected number of hashes that are unused.
  - the probability of a hash collision: 1-(10!/10^10)
  - the expected number of hash collisions: 1-10*(9/10)^10
  - the expected number of hashes that are unused: 10*(9/10)^10
## You call 2 UberX’s and 3 Lyfts. If the time that each takes to reach you is IID, what is the probability that all the Lyfts arrive first? What is the probability that all the UberX’s arrive first?
  - Lyfts arrive first: 2!*3!/5!
  - Ubers arrive first: same
## I write a program should print out all the numbers from 1 to 300, but prints out Fizz instead if the number is divisible by 3, Buzz instead if the number is divisible by 5, and FizzBuzz if the number is divisible by 3 and 5. What is the total number of numbers that is either Fizzed, Buzzed, or FizzBuzzed?
  - 100+60-20=140
## On a dating site, users can select 5 out of 24 adjectives to describe themselves. A match is declared between two users if they match on at least 4 adjectives. If Alice and Bob randomly pick adjectives, what is the probability that they form a match?
  - 24C5*(1+5(24-5))/24C5*24C5 = 4/1771
## A lazy high school senior types up application and envelopes to n different colleges, but puts the applications randomly into the envelopes. What is the expected number of applications that went to the right college?
  - 1
## Let’s say you have a very tall father. On average, what would you expect the height of his son to be? Taller, equal, or shorter? What if you had a very short father?
  - Shorter. Regression to the mean
## What’s the expected number of coin flips until you get two heads in a row? What’s the expected number of coin flips until you get two tails in a row?
## Let’s say we play a game where I keep flipping a coin until I get heads. If the first time I get heads is on the nth coin, then I pay you 2n-1 dollars. How much would you pay me to play this game?
  - less than $3
## You have two coins, one of which is fair and comes up heads with a probability 1/2, and the other which is biased and comes up heads with probability 3/4. You randomly pick coin and flip it twice, and get heads both times. What is the probability that you picked the fair coin?
  - 4/13
## You have a 0.1% chance of picking up a coin with both heads, and a 99.9% chance that you pick up a fair coin. You flip your coin and it comes up heads 10 times. What’s the chance that you picked up the fair coin, given the information that you observed?
  * Events: F = "picked a fair coin", T = "10 heads in a row"
  * (1) P(F|T) = P(T|F)P(F)/P(T) (Bayes formula)
  * (2) P(T) = P(T|F)P(F) + P(T|¬F)P(¬F) (total probabilities formula)
  * Injecting (2) in (1): P(F|T) = P(T|F)P(F)/(P(T|F)P(F) + P(T|¬F)P(¬F)) = 1 / (1 + P(T|¬F)P(¬F)/(P(T|F)P(F)))
  * Numerically: 1/(1 + 0.001 * 2^10 /0.999).
  * With 2^10 ≈ 1000 and 0.999 ≈ 1 this simplifies to 1/2
