# notes from a course 
---
### definition: 
---
what is a probability ? likelihood of an event occuring. This event can be pretty much anything - getting heads , rolling a 4 or even bench pressing 225lbs. We measure probability with numeric values between 0 and 1, because we like to compare the relative likelihood of events. <br>
- Observe the general probabilty formula : P(A) = Preffered / all 
- if two events are independent then P(A and B) = P(A) * P(B)
### Expected value : 
---
- Trial - observing an event occur and recording the outcome
- Example : flip a coin and record the outcome
- Experiment - A collection of one or multiple trials
- Example : flip the coin 20 times and record the outcomes
- Experimental probability - the probability we assign an event , based on an experiment we conduct
- __Expected value__ - the outcome we expected to occur when we run an experiment 
- the expected value can be categorical , numerical , booleane or other depending on the type of the event we are interested in. For instance the expected value of the trial would be the more likely of the two outcomes , where as the expected value of the experiment will be the nimber of time we expect to get either heads or tails after the 20 trials
- Expected value for categorical variables : E(A) = n x P(A)
- Expected value for numeric variables : E(X) = SUM xi x pi
### Probability distribution : 
---
- what is probability distribution ? is a collection of all possible outcome of an event 
- why do we need it ? We need the probability frequency distribution to try and predict future events when the expected value is unattainable.
- what is frequency ? the number of times a given value or outcome appears in the sample space
- what is frequency distribution table ? is a table matching each distinct outcome in the sample space to its associated frequency 

### Complements 
---
the complement of an event is the everything an event is not. We denote the complement of an event with an apostrophe : A' = Not A 
<br>
Characteristics of complements : 
- Can never occur simultaneously
- add up to the sample space : A' + A = sample space 
- Their probabilty add up to 1 
- the complement of a complement is the original event

# Combinatorics :
---
### Permutations : 
---
Permutations represent the number of different possible ways we can arrange a number of elements : <br>
P(n) = n * (n-1) * ... * 1<br>
Characteristics of Permutations : 
- Arranging all elements within the sample space 
- no repetition 
- P(n) = n! called n factorial 
- example : if we need to arrange 5 people we got 5! = 120 ways of doing so.
### Variations : 
---
variations represent the number of different possible ways we can pick and arrange a number of elements.<br>
with repetition : ^V(n , p) = n^p
<br>
whitout repetition : V(n,p)= n! / (n-p)!

### Combinations : 
---
represent the number of different possible ways we can pick a number of elements : C(n,p) = n! / (n-p)! * p!
<br>
Characteristics of combinations : 
- takes into account double counting (selecting john , hanah is the same as selectting hanah , john )
- all the different permutations of a single combination are different variations 
- C = V/P 
- Combinations are symmetric so C(n,p) = C(n , n-p)

### Combinations with separate sample spaces 
--- 
represent the number of different possible ways we can pick a number of elements 
C = n1 x n2 x ... x np <br>
Characteristics of Combinations with separate sample spaces:
-  The option we choose for any element does not affect the number of options for the other elements.
-  The order in which we pick the individual elements is arbitrary.
- We need to know the size of the sample space for each individual element. (𝑛1, 𝑛2 … 𝑛𝑝)

# Bayesian : 
---
### Bayesian notation : 
---
A set is a collection of elements , which hold certain values. Additionally every event has a set of outcomes that satisfy it 
<br>
the null set or empty set is the one that contains nothing .
<br> x ∈ A , we say that x is in A or A contains x (the set in uppercase the element in lower case ) 

### Multiple events : 
---
the sets of outcomes that satisfy two events A adn B can interact in one of the following 3 ways : 
- not touch at all 
- intersect 
- one completely overlaps the other 
### Intersection : 
---
the intersection of two or more events expresses the set of outcomes that satisfy all the events simultaneously.
### Union: 
---
the union of two or more events express the outcomes that satisfy at least one of the events
### Mutually exclusive sets 
---
sets with no overlapping elements are called mutually exclusive. Graphically , their circles never touch.
- remember that all complements are mutually exclusive sets 
### independent and dependent events : 
---
if the likelihood of an event A occuring P(A) is affected by event B occuring , then we say that A and B are dependent events .<br>
if not , the events are independent 
<br>
We express the probability of event A occurring, given event B has occurred the following way 𝑷 𝑨 𝑩 . We call this the conditional probability 
<br>
Independent :
- The outcome of A does not depend on the outcome of B. 
- 𝑃 (𝐴|𝐵) = 𝑃(𝐴)<br>
dependent :
- The outcome of A  depends on the outcome of B. 
- 𝑃 (𝐴|𝐵) != 𝑃(𝐴)

### Conditional probability : 
---
For any two events A and B , such that the likelihood of B occuring is greated than 0 (P(B)>0) , the conditional probability formula states the following : P(A|B) = P(A ∩ B) / P(B) <br>
intuition behind the formula : 
- only interested in the outcome where B is satisfied 
- only the elements in the intersection will satify A as well 
- Parallel to the favoured over all formula : 
	- Intersection = "preferred outcomes"
	- B = sample space
### Law of total probability : 
---
the law of total probability dictates that for any set A , which is a union of manu mutually exclusive sets B1 , .... , Bn , its probability equals the following sum : 𝑃 (𝐴) = 𝑃 (𝐴|𝐵1) × 𝑃 (𝐵1) + ⋯ + 𝑃 (𝐴|𝐵𝑛) × 𝑃(𝐵𝑛)
<br>
### Bayes law : 
---
Bayes’ Law helps us understand the relationship between two events by computing the different conditional probabilities. We also call it Bayes’ Rule or Bayes’ Theorem <br>
𝑃 (𝐴|𝐵) = 𝑃 (𝐵|𝐴) × 𝑃 (𝐴) / 𝑃(B) 

# Distributions : 
---
### An overview : 
---
A distribution shows the possible values a random variable can take and how frequently they occur. <br>
We call a function that assigns a probability to each distinct outcome in the sample space, a probability function.

### Types of distribution : 
---
Certain distributions share characteristics, so we separate them into types. The well-defined types of distributions we often deal with have elegant statistics. We distinguish between two big types of distributions based on the type of the possible values for the variable – discrete and continuous.
__Discrete:__
____
- Have a finite number of outcomes.
- Can add up individual values to determine probability of an interval.
- Can be expressed with a table, graph or a piece-wise function. 
- Expected Values might be unattainable.
- Graph consists of bars lined up one after the other.
__Continuous__ : 
--- 
- Have infinitely many consecutive possible values. 
-  Use new formulas for attaining the probability of specific values and intervals. 
- Cannot add up the individual values that make up an interval because there are infinitely many of them. 
- Can be expressed with a graph or a continuous function. 
- Graph consists of a smooth curve.

### Discrete Distributions: 
---
- A distribution where all the outcomes are equally likely is called a Uniform Distribution.
- A distribution consisting of a single trial and only two possible outcomes – success or failure is called a Bernoulli Distribution.
- A sequence of identical Bernoulli events is called Binomial and follows a Binomial Distribution
- When we want to know the likelihood of a certain event occurring over a given interval of time or distance we use a Poisson Distribution.
### Continuous Distributions: 
---
- A Normal Distribution represents a distribution that most natural events follow.
- To standardize any normal distribution we need to transform it so that the mean is 0 and the variance and standard deviation are 1.
- Students’ T Distribution : A Normal Distribution represents a small sample size approximation of a Normal Distribution.
- A Chi-Squared distribution is often used.
- The Exponential Distribution is usually observed in events which significantly change early on.
- The Continuous Logistic Distribution is observed when trying to determine how continuous variable inputs can affect the probability of a binary outcome