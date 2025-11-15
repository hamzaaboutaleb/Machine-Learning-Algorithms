### statistic and probability :
---
- What is Marginal probability ? <br>
it is simply the chance of one specific event happening without worrying about what happens with other events . for example if you re looking at the probability of it raining tomorrow , you only care about the chance of rain not what happens with other weather canditions like wind or temperature .

- What are the probability Axioms ? <br>
are just basic rules that help us undestand how probabilities work . there are three main ones : 
1. Non negativity axiom : Probability cant be negative , always between 0 and 1 
2.  Normalization axiom : if something is certain to happen , its probability is one 
3. Additivity axiom : if two events cant happen at the same time (like rolling a 3 or a 4 on a die) the chance of eather one happening is the sum of their individual chances 

- whats the difference between dependent and independent events in probability ? 
independent events : two events are independent if one event doesnt change the likelihood of the other happening . for example , flipping a coin twice - the first flip doesnt affect the second flip so the proba of both events happening is jujt the product of their individual probabilities Dependent events : if one affect the likelihood of the other event. if u draw a card from a deck and dont put it back , the chance of a second card depends on what the first card was. 

- What is canditional probability ? 
refers to the probability of an event occuring given that an event already occured. Mathematically , it is defined as the probability of event A occuring , giving that event B has occured and is denoted P(A|B) . the formula for canditional probability : 
P(A∣B)=P(A∩B)​/P(B)

- what is Bayes Theorem and when we use it in data science ? 
___
Bayes Theorem helps us figure out the probability of an event happening based on some prior knowkedge or evidence.  Its like updating our guess about something when we learn new things , the formula for Bayes Theorem is : 
P(A∣B)=P(B∣A)⋅P(A)​/P(B)

- Define variance and canditional variance : 
___
variance is a way to measure how spread out or different the numbers in a dataset are from the __average__, the variance is low. If the numbers are all close to the average , the variance is low , if the numbers are spread far apart , the variance is high.<br>
canditional variance is similar but it looks at how much a variable changes when we know something else about is. For example, imagine you want to know how much people's height varies based on their age. The conditional variance would tell you how much height changes for a specific age group, using the knowledge of age to focus on the variability within that group.

- Explain the concepts of Mean , Median , Mode and standard deviation :
---
Mean : the mean is simply the average of a set of numbers , to find it , you add up all the numbers and divide by how many numbers there are. It gives you a central value that represents the overall data.
<br>
Median : The median is the middle number when you arrange the data in order from smallest to largest. If there’s an even number of numbers, you average the two middle numbers. The median is useful because it’s not affected by extremely high or low values, making it a better measure of the "middle" when there are outliers.<br>

Mode: The mode is the number that appears the most often in your data. You can have one mode, more than one mode or no mode at all if all the numbers appear equally often.<br>
Standard Deviation: Standard deviation tells us how spread out the numbers are. If the numbers are close to the average, the standard deviation is small. If they’re more spread out, it’s large. It shows us how much variation or "scatter" there is in the data. is the square root of the variance so the std is easy to interpret that the variance.

- what is the normal distribution and standard normal distribution ? 
---
Normal Distribution : A normal distribution is a bell-shaped curve that shows how most data points are close to the average(mean)and the further away you go from the mean , the less likely those points are. its a common pattern in nature like people's heights or test scores. <br>
Standard Normal Distribution : this is a special type of normal distribution where the mean is 0 and the standard deviation is 1. it helps make comparaisons between different sets of data easier because the different sets of data easier because the data is standardized.

- what is the difference between correlation and causation : 
- ---
correlation means that tyat two things are related or happen at the same time but one doesnt necessarily cause the other . eating ice cream and swim example <br>
causation means one thing directly causes the other to happen . For example studying and have good score.

- What are Uniform, Bernoulli and Binomial Distributions and how do they differ?
- ---
Uniform distribution : means that every possible outcome has an equal chance of occuring . when rolling a fair six sided die each number has the same probability of showing up , resulting in a flat line when graphed 
<br>
Bernoulli Distribution  : is uded in situations where there are only two possible outcome such as success or failure , a common example is flipping a coin where you either get heads or tails .
<br>
Binomial Distribution : applies when you perform a set number of independent trials, each with two possible outcomes. It helps calculate the probability of getting a specific number of successes across multiple trials such as flipping a coin 5 times and determining the chance of getting exactly 3 heads.

- explain the exponential distribution and where its commonly used 
---
it helps us understand the time between random events that happen at a constant rate. For example, it can show how long you might have to wait for the next customer to arrive at a store or how long a light bulb will last before it burns out.

- Desribe the Poisson Distribution and its characteristics : 
---
the poison distribution tells us how often an event happens within a certain period of time or space. Its used when events happen at a steady rate like how many cars pass by a toll booth in an hour . <br> key points : 
1. it counts the number of events that happen. 
2. the events happen at a constant rate . 
3. Each event is independent meaning one event doesnt affect the others

- Explain the t-distribution and its relationship with the normal distribution 
- ---
is similar to the normal distribution but its used when er dont have much data and dont know the exact spread of the population. its wider and more spread out than the normal distribution but as we get more data it looks like the normal distribution.

- Describe the chi squared distribution : 
- --- 
used when we want to test how well our data matches a certain pattern or to see if two things are related. its often used in tests like checking if dice rolls are fair or if two factors like age and voting preference are linked

- what is the difference between z-test , F-test and t-test ? 
---
z-test : we use the z test when we want to compare the average of a sample to a known average of a larger population and we know the population's spread (standard deviation). It’s typically used with large samples or when we have good information about the population.<br>
**T-test:** The t-test is similar to the z-test, but it's used when we don't know the population’s spread (standard deviation). It’s often used with smaller samples or when we don’t have enough data to know the population’s spread.<br>
***F-test:*** The F-test is used when we want to compare how much the data is spread out (variance) in two or more groups. For example, you might use it to see if two different teaching methods lead to different results in students.<br>

- what is the central limit theorem and why is it significant in statistics ? 
- ---
The Central limit theorem says that if you take many samples from a population, no matter how the population looks, the average of those samples will start to look like a normal (bell-shaped) distribution as the sample size gets bigger. This is important because it means we can use normal distribution rules to make predictions, even if the population itself doesn’t look normal.

- Describe the process of hypothesis testing including null and alternative hypotheses :
----
Hypothesis testing helps us decide if a claim about a population is likely to be true, based on sample data.

- Null Hypothesis (H0): This is the "no effect" assumption, meaning nothing is happening or nothing has changed.
- Alternative Hypothesis (H1): This is the opposite, suggesting there is a change or effect.

We collect data and check if it supports the alternative hypothesis or not. If the data shows enough evidence, we reject the null hypothesis.

- How do you calculate a confidence interval and what does it represent ? 
---
A confidence interval gives us a range of values that we believe the true population value lies in , based on our sample data. <br>
to calculate : you first collect sample data , then calculate the sample mean and margin of error (how much the sample result could vary) . the confidence interval is the range around the mean where the true population value should be , with a certain level of confidence (like 95%)

- what is a p-value in statistics : 
----
A p-value tells us how likely it is that we would get the data we have if the null hypothesis were true . A small p-value (less than 0.05) means the data is unlikely under the null hypothesis, so we may reject the null hypothesis. A large p-value means the data fits with the null hypothesis, so we don’t reject it. <br>
A **p-value** answers this question:

“If the null hypothesis were true, how surprising is my data?”

- Explain type I and Type II errors in hypothesis testing : 
- ---
type I error (False Positive) : Mistakenly reject a true null hypothesis , thinking something has changed when it hasnt.
<br>
type II error (False Negative) : fail to reject a false null hypothesis ,missing a real effect

- What is the significance leve (alpha) in hypthesis testing ? 
- ---
 The ****significance level (alpha)**** is the threshold you set to decide when to reject the null hypothesis. It shows how much risk you're willing to take for a Type I error (wrongly rejecting the null hypothesis). Commonly, alpha is 0.05, meaning there’s a 5% chance of making a Type I error.

- How can you calculate the correlation coefficient between two variables ? 
- --
The correlation coefficient measures how strongly two variables are related.

To calculate it, you:

1. Collect data for both variables.
2. Find the average for each variable.
3. Calculate how much the variables move together (covariance).
4. Divide by the standard deviations to standardize the result.

This gives you a number between -1 and 1 where 1 means a perfect positive relationship, -1 means a perfect negative relationship and 0 means no relationship.

- what is covariance and how is it related to correlation ? 
- ---
- ****Covariance**** shows how two variables change together. If both increase together, covariance is positive and if one increases while the other decreases, it’s negative. However, it depends on the scale of the variables, so it's harder to compare across different data.
- ****Correlation**** standardizes covariance by using the standard deviations of the variables. It’s easier to interpret because it gives you a number between -1 and 1 that shows the strength and direction of the relationship.

- explain how to perform a hypothesis test for comparing two population means ? 
---
When comparing two population means, we:

1. Set up hypotheses:

	- Null hypothesis (H0): The two means are equal.
	- Alternative hypothesis (H1): The two means are different.
	- Collect data from both populations.

2. Calculate the test statistic (often using a t-test or z-test).

3. Compare the results to see if the difference is statistically significant.

4. If the results show a big enough difference, we reject the null hypothesis.

- explain multivariate distribution in data science : 
---
A multivariate distribution involves multiple variables and it helps us model situations where we care about the relationships between those variables. For example, predicting house prices based on factors like size, location and age of the house. It’s a way to see how different features or variables work together and affect the outcome.

- Describe the concept of conditional probability density function (PDF) : 
---
A PDF describes the probability of an event happening, given that we already know some other event has occurred. For example, it tells us the chance of a person getting a disease given they have a certain symptom. It helps us understand how one event affects the probability of another.

- What is the cumulative distribution function (CDF) and how is it related to PDF?
- --
The probability that a continuous random variable will take on particular values within a range is described by the Probability Density Function (PDF), whereas the CDF provides the cumulative probability that the random variable will fall below a given value. Both of these concepts are used in probability theory and statistics to describe and analyse probability distributions. The PDF is the CDF’s derivative and they are related by integration and differentiation.

-  What is ANOVA? What are the different ways to perform ANOVA tests?
--- 
The statistical method known as ****ANOVA**** or ****Analysis of Variance****, is used to examine the variation in a dataset and determine whether there are statistically significant variations between group averages. When comparing the means of several groups or treatments to find out if there are any notable differences, this method is frequently used.

There are several different ways to perform ANOVA tests, each suited for different types of experimental designs and data structures:

1. [****One-Way ANOVA****](https://www.geeksforgeeks.org/machine-learning/one-way-anova/)
2. [****Two-Way ANOVA****](https://www.geeksforgeeks.org/maths/two-way-anova/)

When conducting ANOVA tests we typically calculate an F-statistic and compare it to a critical value or use it to calculate a p-value.

- What is the difference between descriptive and inferential statistics ? 
---
Descriptive Statistics aims to summarize and present the features of a given dataset , while inferential statistics leverages sample data to make estimates or test hypotheses about a larger population .<br>
Descriptive statistics : 
it describe the key aspects or characteristics of a dataset : 
1. Measures of Central tendecy