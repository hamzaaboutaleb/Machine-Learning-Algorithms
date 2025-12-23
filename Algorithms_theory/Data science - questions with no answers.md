
### Questions 
---

1. what is data science ? 
2. what is marginal probability ? 
3. what are the probability axioms ? 
4. what is the difference between dependent and independent events in probability ? 
5. what is conditional probability ? 
6. what is bayes theorem and when do we use it in data science ? 
7. define variance and conditional variance ? 
8. explain the concept of the mean , median , mode and standard deviation ?
9. what is normal distribution and standard normal distribution ? 
10. what is the difference between correlation and causation ? 
11. what are uniform,bernoulli and binomial distributions and how do they differ ? 
12. explain the exponential distribution and where its commonly used ?
13. Describe the Poisson distribution and its characteristics ? 
14. Explain the t-distribution and its relationship with the normal distribution ? 
15. Describe the chi-squared distribution ? 
16. what is the difference between z-test , F-stest and t-test ? 
17. what is CLT ? 
18. Describe the process of hypothesis testing including null and alternative hypotheses ? 
19. How do we calculate a confidence interval and what does it represent ? 
20. what is a p-value in statistics ? 
21. explain the Type I and Type II errors in hypothesis testin ? 
22. what is the significance level (alpha) in hypothesis testing ? 
23. how can you calculate the correlation coefficient between two variables ? 
24. what is covariance and how is it telated to correclation ?
25. explain how to perform a hypothesis test for comparing two population means ? 
26. explain multivariate distribution in data science  ? 
27. explain the concept of conditional probability density function PDF  ? 
28. what is the cumulative distribution function and how is it related to PDF ? 
29. What is ANOVA ? what are the different ways to perform ANOVA tests ? 
30. what is the difference between a population and a sample in statistics ? 
31. Explain the A/B testing and its application ? what are some common pitfalls encoutered in A/B testing ? 
32. Describe the hypothesis testing and p-value in layman's term ? and give a pratival application to them ? 
33. what is the meaning of selection bias and how to avoid it ? 

## Responses : 
---

### 1. what is data science ? 
---
- A field that extracts knowledge and insights from structured and unstructered data by using scientific methods , algorithms , processes and systems. It combines expertise from various domains such as statistics , computer science , machine learning , data engineering and domain specific knowledge to analyze and interpret complex data sets . 
- Un domaine qui extrait des connaissances et des informations à partir de données structurées et non structurées à l'aide de méthodes scientifiques, d'algorithmes, de processus et de systèmes. Il combine l'expertise de divers domaines tels que les statistiques, l'informatique, l'apprentissage automatique, l'ingénierie des données et les connaissances spécifiques à un domaine afin d'analyser et d'interpréter des ensembles de données complexes. 

### 2. What is Marginal probability ? 
---
- Marginal Probability is simply the chance of one specific event happening, without worrying about what happens with other events. For example, if you’re looking at the probability of it raining tomorrow, you only care about the chance of rain, not what happens with other weather conditions like wind or temperature.
- La **probabilité marginale** est simplement la chance qu’un événement spécifique se produise, sans se préoccuper de ce qui arrive aux autres événements.  Par exemple, si vous regardez la probabilité qu’il pleuve demain, vous vous intéressez uniquement à la chance de pluie, et non à ce qui se passe avec d’autres conditions météorologiques comme le vent ou la température.
### 3. what are the probability axioms ? 
---
The ***probability axioms** are just basic rules that help us understand how probabilities work. There are three main ones:

1. ***Non-Negativity Axiom:** Probabilities can't be negative. The chance of something happening is always 0 or more, never less.
2. **Normalization Axiom:*** If something is certain to happen (like the sun rising tomorrow), its probability is 1. So, 1 means "definitely happening."
3. ***Additivity Axiom:** If two events can't happen at the same time (like rolling a 3 or a 4 on a die), the chance of either one happening is just the sum of their individual chances.

se sont les regles fondamentales qui definissent une probabilite . ils ont ete formalises par kolmogorov , voici les trois principaux axiomes : 
1. Non negativite : Pour un evenement A ,sa probabilite est toujours positive ou nulle 
2. Probabilite de l univers : la probabilite de l ensemble de tout les evenements possibles est egale a 1 
3. Additivite : si deux evenements A et B ne peuvent par se produire en meme temps , alors la probabilte que l un ou l autre se produise est la somme de leurs probabilites . 

### 4. what is the difference between dependent and independent events in probability ? 
---
- ***Independent Events:** Two events are independent if one event doesn't change the likelihood of the other happening. For example, flipping a coin twice – the first flip doesn't affect the second flip. So, the probability of both events happening is just the product of their individual probabilities.
- ***Dependent Events:** Two events are dependent if one event affects the likelihood of the other happening. For example, if you draw a card from a deck and don't put it back (without replacement), the chance of drawing a second card depends on what the first card was. The probability changes because one card was already taken out.
- deux evenements A et B sont independants si la survenue de l un n affecte pas la probabilite de l autre

###  4. What is Conditional Probability?
---
- it refers to the probability of an event occurring given that another event has already occurred.
- La probabilite conditionnelle decrit la probabilite qu un evenement A se produise sachant qu un autre evenement B est deja survenu. 

### 5. What is Bayes’ Theorem and when do we use it in Data Science?
---
- it helps us figure out the probability of an event happening based on some prior knowledge or evidence. It’s like updating our guess about something when we learn new things.
- Le théorème de Bayes permet de **mettre à jour la probabilité d’un événement A en fonction d’une nouvelle information B**.

### 6. Define Variance and Conditional Variance.
---
- Variance is a way to measure how spread out or different the numbers in a dataset are from the average. If the numbers are all close to the average, the variance is low. If the numbers are spread far apart, the variance is high. Think of it like measuring how much everyone’s test score differs from the average score in a class.
- Conditional variance is similar, but it looks at how much a variable changes when we know something else about it. For example, imagine you want to know how much people's height varies based on their age. The conditional variance would tell you how much height changes for a specific age group, using the knowledge of age to focus on the variability within that group.

### 7. Explain the concepts of Mean, Median, Mode and Standard Deviation.
---
- the mean is simply the average of a set of numbers. To find it, you add up all the numbers and divide by how many numbers there are. It gives you a central value that represents the overall data . 
- The median is the middle number when you arrange the data in order from smallest to largest. If there’s an even number of numbers, you average the two middle numbers. The median is useful because it’s not affected by extremely high or low values, making it a better measure of the "middle" when there are outliers.
- The mode is the number that appears the most often in your data. You can have one mode, more than one mode or no mode at all if all the numbers appear equally often.
- Standard deviation tells us how spread out the numbers are. If the numbers are close to the average, the standard deviation is small. If they’re more spread out, it’s large. It shows us how much variation or "scatter" there is in the data.
### 8. What is Normal Distribution and Standard Normal Distribution?
---
- A normal distribution is a bell-shaped curve that shows how most data points are close to the average (mean) and the further away you go from the mean, the less likely those data points are. It’s a common pattern in nature like people's heights or test scores.
- la distribution normale est une distribution de probabilite continue qui a la forme d une cloche symetrique autour de sa moyen 
- This is a special type of normal distribution where the mean is 0 and the standard deviation is 1. It helps make comparisons between different sets of data easier because the data is standardized.

### 9. What is the difference between correlation and causation?
---
- Correlation means that two things are related or happen at the same time, but one doesn’t necessarily cause the other. For example, if people eat more ice cream in summer and also go swimming more, there's a correlation between the two, but eating ice cream doesn’t cause swimming. They just both happen together.
- Causation means one thing directly causes the other to happen. For example, if you study more, your test scores will likely improve. In this case, studying causes better test scores. To prove causation, you need more evidence, often from experiments, to show that one thing is actually causing the other.

### 10. What are Uniform, Bernoulli and Binomial Distributions and how do they differ?
---
- Uniform distribution means that every possible outcome has an equal chance of occurring. For example, when rolling a fair six-sided die, each number (1 through 6) has the same probability of showing up, resulting in a flat line when graphed.
- Bernoulli distribution is used in situations where there are only two possible outcomes such as success or failure. A common example is flipping a coin where you either get heads (success) or tails (failure).
- Binomial distribution :applies when you perform a set number of independent trials, each with two possible outcomes. It helps calculate the probability of getting a specific number of successes across multiple trials such as flipping a coin 5 times and determining the chance of getting exactly 3 heads.

### 11. Explain the Exponential Distribution and where it’s commonly used.
---
it helps us understand the time between random events that happen at a constant rate. For example, it can show how long you might have to wait for the next customer to arrive at a store or how long a light bulb will last before it burns out.

### 12. Describe the Poisson Distribution and its characteristics.
---
it tells us how often an event happens within a certain period of time or space. It’s used when events happen at a steady rate like how many cars pass by a toll booth in an hour.

Key points:

- It counts the number of events that happen.
- The events happen at a constant rate.
- Each event is independent, meaning one event doesn’t affect the others.
### 13. Explain the t-distribution and its relationship with the normal distribution.
---
it is similar to the normal distribution, but it’s used when we don’t have much data and don’t know the exact spread of the population. It’s wider and more spread out than the normal distribution, but as we get more data, it looks more like the normal distribution.

### 14. Describe the chi-squared distribution.
---
it s used when we want to test how well our data matches a certain pattern or to see if two things are related. It’s often used in tests like checking if dice rolls are fair or if two factors like age and voting preference, are linked.

### 15. What is the difference between z-test, F-test and t-test?
---
- z-test : We use the z-test when we want to compare the average of a sample to a known average of a larger population and we know the population's spread (standard deviation). It’s typically used with large samples or when we have good information about the population.
- The t-test is similar to the z-test, but it's used when we don't know the population’s spread (standard deviation). It’s often used with smaller samples or when we don’t have enough data to know the population’s spread.
- The F-test is used when we want to compare how much the data is spread out (variance) in two or more groups. For example, you might use it to see if two different teaching methods lead to different results in students.
### 16. What is the central limit theorem and why is it significant in statistics?
---

it says that if you take many samples from a population, no matter how the population looks, the average of those samples will start to look like a normal (bell-shaped) distribution as the sample size gets bigger. This is important because it means we can use normal distribution rules to make predictions, even if the population itself doesn’t look normal.


### 17. Describe the process of hypothesis testing , including null and alternative hypotheses ? 
--- 
Hypothesis testing helps us decide if a claim about a population is likely to be true, based on sample data.

- Null Hypothesis (H0): This is the "no effect" assumption, meaning nothing is happening or nothing has changed.
- Alternative Hypothesis (H1): This is the opposite, suggesting there is a change or effect.

We collect data and check if it supports the alternative hypothesis or not. If the data shows enough evidence, we reject the null hypothesis.
### 18. How do you calculate a confidence interval and what does it represent?
---
A confidence interval gives us a range of values that we believe the true population value lies in, based on our sample data.

To calculate: You first collect sample data, then calculate the sample mean and margin of error (how much the sample result could vary). The confidence interval is the range around the mean where the true population value should be, with a certain level of confidence (like 95%).

### 19. What is a p-value in statistics?
---
A p-value tells us how likely it is that we would get the data we have if the null hypothesis were true. A small p-value (less than 0.05) means the data is unlikely under the null hypothesis, so we may reject the null hypothesis. A large p-value means the data fits with the null hypothesis, so we don’t reject it.

### 20. Explain Type I and Type II errors in hypothesis testing.
---
- Type I error (False positive) :Mistakenly reject a true null hypothesis, thinking something has changed when it hasn’t.
- - Type II error (False negative) Fail to reject a false null hypothesis, missing a real effect.

### 21. What is the significance level (alpha) in hypothesis testing?
---
the significance level alpha is the threshold you set to decide when to reject the null hypothesis. It shows how much risk you re willing to take for a Type I error (wrongly rejecting the null hypothesis) . commonly alpha is 0.05 , meaning theres 5% chance of making type I error.

### 22. How can you calculate the correlation coefficient between two variables?
---
1. Collect data for both variables 
2. find the average of each variable 
3. calculate how much the variables move togethet (covariance) 
4. divide by standard deviations to standardize the result 


this gives you a number between -1 and 1 where 1 means a perfect positive relationship , .1 means a perfect negative relationship and 0 means no relationship. 

### 23. what is covariance and how is it related to correlation ? 
---
- covariance shows how two variables change together. If both increase together , covariance is positive and if one increases while the other decreases , its negative. however it depends on the scale of the variables , so its harder to compare accross different data 
- correlation standardize covariance by using standard deviations of the variables . Its easier to interpret because it gives you a number between -1 and 1 that shows the strenght and direction of the relationship

### 24. Explain how to perform a hypothesis test for comparing two population means.
---
when comparing two population means , we : 
1. Set up hypotheses : 
	- null hypothesis (H0) : the two means are equal. 
	- Alternative one H1 : the two means are different 
	- collect data from both populations 
2. calculate the test statistic (often using a t-test or z-test) 
3. compare the results to see if the difference is statistically significant 
4. if the results show a big enough difference  , we reject the null hypothesis 

### 25. Explain multivariate distribution in data science.
---
A multivariate distribution involves multipe variables and it helps us model situations where we care about the relationships between those variables. for example predicting house prices based on factors like size , location and age of the house . its a way to hee how different features or variables work together to affect the outcome.

### 26. describe the concept of conditional probability density function : PDF 
---
it describes the probability of an event happening, given that we already know some other event has occurred. For example, it tells us the chance of a person getting a disease given they have a certain symptom. It helps us understand how one event affects the probability of another.

### 27. What is the cumulative distribution function (CDF) and how is it related to PDF?
---
the probability that a continuous random variable will take on particular values within a range is described by the probabilty density function whereas the CDF provides the cumulative probability that the random variable will fall below a give value. Both of these concepts are used in probability theory and statistics to describe and analyse probability distributions. The PDF is the CDF’s derivative and they are related by integration and differentiation.

### 28. What is ANOVA? What are the different ways to perform ANOVA tests?
---
The statistical method known as ****ANOVA**** or ****Analysis of Variance****, is used to examine the variation in a dataset and determine whether there are statistically significant variations between group averages. When comparing the means of several groups or treatments to find out if there are any notable differences, this method is frequently used.

There are several different ways to perform ANOVA tests, each suited for different types of experimental designs and data structures:

1. [****One-Way ANOVA****](https://www.geeksforgeeks.org/machine-learning/one-way-anova/)
2. [****Two-Way ANOVA****](https://www.geeksforgeeks.org/maths/two-way-anova/)

When conducting ANOVA tests we typically calculate an F-statistic and compare it to a critical value or use it to calculate a p-value.
### 29. What is the difference between a population and a sample in statistics?
---
- ****Population:**** This is the whole group you want to study. For example, if you're looking at the average height of all students in a school, the population is every student in that school.
- ****Sample:**** A sample is just a smaller part of the population. Since it's often not possible to study everyone, you choose a few people from the group to represent the whole population. For example, you might measure the height of 100 students and use that data to estimate the average height of all students.

### 30 . define mean , median , mode , range : 
---
- mean : is the average of numbers 
- median : is the value in the middle of a sorted list 
- mode : the most frequently occuring value 
- Range : is the difference between the max and the min values in a dataset. it gives an idea of data spread but can be affected by outliers 
- Variance : measure how far the data point are spreadout from the mean . it is calculated as the average. It is calculated as the average of squared deviations from the mean. A high variance means data points are widely spread
- standard deviation : the square root of variance , showing the average distance of values from the mean, Unlike variance , it is in the same unit as the data
- outliers : are extreme values that differ significantly from the rest of the data, they can be detected using statistical methods like the IQR rute or standard deviation 

### 31. difference between Descriptive and inferential statistics : 
---
- Descriptive summarize and describe the data 
- Inferential : use sample data to make predictions or generalizations about a population 

### 32. define independent and dependent events in probability : 
---
- Independent events : the outcome of one event does not affect the other 
- Dependent events : the outcome of one event affects the other 



