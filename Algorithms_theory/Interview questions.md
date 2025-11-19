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
__Descriptive statistics :__
it describe the key aspects or characteristics of a dataset : 
1. Measures of Central tendency : identify central or typical values in the dataset typically using the mean , median or mode .
2. Measures of spread or dispersion : indicate the variability or spread around the central value , often quantified by the range , standard deviation or variance . 
3. Data distribution : Categorized the data distribution as normal , skewed or otherwise and assists in visual presentation 
4. Shape of data : describes wheter the data is symmetrical or skewed and the extent of that skewness 
5. Correlation : Measures the relationship or lack thereof between two variables 
6. Text statistics : summarizes verbal or written data using word frequencies , readabilities , etc. <br>
__Inferential Statistics__ :
in constrast inferential statistics extends findings from a subset of data to inferences about an entire population . 
- **Hypothesis Testing**: Allows researchers to compare data to an assumed or expected distribution, indicating whether a finding is likely due to chance or not.
- **Confidence Intervals**: Provides a range within which the true population value is likely to fall.
- **Regression Analysis**: Predicts the values of dependent variables using one or more independent variables.
- **Probability**: Helps measure uncertainty and likelihood, forming the basis for many inferential statistical tools.
- **Sampling Techniques**: Guides researchers in selecting appropriate samples to generalize findings to a wider population.

---

## Machine Learning interview questions : 
---
1. __What do you understand by ML , and how does it differ from ai and data science ?__
---
ML is a branch of artificial intelligence that deals with building algorithms capable of learning from data . instead of being programmed with fixed rules , these algorithms identify patterns in data and use them to make predictions or decisions that improve with experience.

2. __What is overfitting in ML , and how can it be avoid ?__
---
it occurs when a model not only learns the true patterns in the training data but also memorizes the noise or random fluctuations. this results in high accuracy on training data but poor performance on unseen/test data.<br> 
ways to avoid it : 
- __Early stopping__ : Strop training when validation accuracy stops improving , even if training accuracy is still increasing . 
- __Regularization__ : apply techniques like L1 (Lasso) or L2 (Ridge) Regularization with add penalties to large weights to reduce model complexity. 
- __Cross-validation__ : use k-fold cross validation to ensure the model generalizes well.
- __Dropout (for neural networks)__ : Randomly drop neurons during training to prevent over-reliance on specific nodes .
- __simpler models__ : avoid overly complex models when simpler ones can explain the data well .

3. Undefitting ? 
---
it occurs when a model is too simple to capture the underlying patterns in the data. this leads to poor accuracy on both training and test data.
<br>
_ways to avoid it :_ 
- use a more complex model : chose a model with higher complexity to learn patten like decision trees, neural network ... 
- add relevant features : include meaningful features that better represent the data 
- reduce regularization : too much regularization can restrict the models ability to learn 
- train longer : allow the model more epochs or iterations to properly learn patterns 

4. __what is Regularization ?__ 
---
is a technique used to reduce model complexity and prevent overfitting. It works by adding a penalty term to the loss function to discourage the model from assigning too much importance(large weights) to specific features. This helps the model generalize better on unseen data.<br>
ways to apply regularization :
- ***L1 Regularization (Lasso):** Adds the absolute value of weights as a penalty which can shrink some weights to zero and perform feature selection.
- ***L2 Regularization (Ridge):** Adds the squared value of weights as a penalty which reduces large weights but doesn’t eliminate them.
- ***Elastic Net:** Combines both L1 and L2 penalties to balance feature selection and weight reduction.
- **Dropout (for Neural Networks)** Randomly drops neurons during training to avoid over-reliance on specific nodes.

5. __Explain Lasso and ridge regularization, How do they help elastic net regularization ?__
---
- __Lasso Regularization(L1)__ : Lasso adds a penalty equal to the absolute values of the models weights to the loss function. it can shrink some weights to exaclty zero, performing feature selection.
- __Ridge regularization__ : it adds a penalty equal to the square of the models weights to the loss function . it reduces large wights but does not set them to zero, helping generalization <br>
__Key differences__ : 
	 - ***Lasso (L1):** Can set weights to zero → feature selection. Use it when we have many irrelevant features.
	- ***Ridge (L2):** Reduces weights but keeps all features → no feature elimination. Use when all features are useful but want to avoid overfitting.
-__Elastic net regularization__ : combines both l1 and l2 penalties , balancing feature selection and weight reduction. It is especially useful when features are correlated , as it avoids lasso's limitation of picking only one feature from a group .

5. __What are different model evaluation techniques in machine learning ?__
---
Model evaluation techniques are used to assess how well a machine learning model performs on unseen data, 
choosing the right technique depends on the type of problem like classification , regression etc and type of dataset we have.
- Train-test split : divide data into training and testing sets like 70:30 or 80:20 to evaluate model performance on unseen data. here 70% data will be used for training and 30% will be used to test accuracy of model.
- cross-validation : split data into k folds , train on k-1 folds validate on the remaining fold and average the results to reduce bias 
- Confusion matrix (for classification) : counts True Positives , True Negatives ,False Positives and False Negatives 
- Accuracy: Proportion of correct predictions over total predictions 
- Precision : here correct positive predictions are divided by total predicted positives 
- Recall : correct positive predictions are divided by total actual positives 
- F1 . score : harmonic mean of precision and recall. It balances precision and recall 
- ROC curve & AUC : measures models abulity to distinguish between classes , here AUC is area under the ROC curve
- Loss Functions (for regression , classification) : quanitifies prediction error to optimize model . It can include : mean absolute error , mean squared error .. .

7. __Explain confusion matrix :__ 
---
is a table used to evaluate classification model. It compares the predicted labes with the actuals labels telling how well the model is performing and what type of errors it makes.

8. __What is the difference between precision and recall ? how F1 combines both ?__ 
---
Precision : it is the ration between the true positives and all the positive examples predicted by the model. in other words , precision measures how many of the predicted positive examples are actually true positives. it is a measure of the model's ability to avoid false positives and make accurate positive predictions. <in spam detection , high precision means most emails marked as spam are true spam>
<br>
Recall : it calculate the ration of true positives and the total number of examples that actually fall in the positive class . Recall measures how many of the actual positive examples of the models ability to avoid false negatives and identify all positive examples correctly .
<In disease detection , high recall means most sick patients are correctly identified><br>
Precision is about being exact(avoiding false positive)
, recall is about being comprehensive (avoiding false negatives)

9. __Different Loss Functions in machine learning :__  
---
Loss functions measure the error between the model's predicted output and the actual target value. They guide the optimization process during training. Some of them are : <br>
- Mean Squared Error (MSE) : used in regression problem. It penalizes larger errors more heavily by squaring them 
- Mean Absolute Error (MAE) : used in regression as it takes absolute differences between predicted and actual values. it is less sensitive to outliers than MSE
- Huber loss : It combines MSE and MAE making it less sensitive to outliers than MSE.
- Cross-Entropy Loss (Log loss):Used in classification problem. It measures the difference between predicted probability distribution and actual labels.
- Hinge Loss :  Used for classification with SVMs. It encourages maximum margin between classes.
- KL divergence : measures how one probability distribution differs from another hence used in probabilistic models. 
- Exponential loss : used in boosting methids like AdaBoost; penalizes misclassified points more strongly
- R-squared (R^2) : used in regression and measures how well the model explains variance in the target variable

10. __What is AUC-ROC curve ?__  
---
ROC curve(receiver operating characteristic) : the ROC curve is a graphical plot that shows the trade-off between true positive RATE (TPR/Recall) and False Positive Rate at different threshold values .
AUC(area under the curve) : auc is the area under the ROC curve. it represents the probability that a radomly chosen positive instance is ranked higher than a randomly chosen negative instance. 
- AUC = 1 -> perfect classifier
- AUC = 0.5 -> Random guessing 
- AUC < 0.5 -> worse than random 
Roc shows performance across thresholds , AUC summarizes overall model performance into a single number <br>
***Example:** If a medical test has an AUC of 0.90, it means there’s a 90% chance that the model will rank a randomly chosen diseased patient higher than a healthy one.

11. __is accuracy always a good metric for classification performance ?__  
---
No, accuracy can be misleading, especially with imbalanced datasets. In such cases:

- Precision and Recall provide better insight into model performance.
- F1-score combines precision and recall as their harmonic mean, giving a balanced measure of model effectiveness, especially when the classes are imbalanced.

12. __What is cross-validation ?__  
---
Cross-validation is a model evaluation technique used to test how well a machine learning model generalizes to unseen data. Instead of training and testing on a single split, the dataset is divided into multiple subsets (called folds) and the model is trained and tested multiple times on different folds. <br> <br>
how It works :  
1. Split the dataset into k folds like 5 or 10
2. train the model on (k-1) folds and test it on the remaining fold
3. repeat this process k times so that every fold is used for testing once 
4. take the average of all results as the final performace score . <br>
Types of Cross-validation : 
- K-fold cross validation : dataset is divided into k equal fold and training/testing is repeated k times .
- stratified k-fold : similar to k-fold but keeps class distribution balances(useful in classification)
- Leave-one-out(LOO) : special case where k = number of samples and every single point acts as a test set once
- Hold-out method : simple train/test split and is considered a basic form of validation

1. __Explain k-fold Cross-validation, leave one out and hold-out method :__  
---
- K-Fold Cross-Validation : the dataset is divided into k equal folds. The model is trained on (k-1) folds and tested on the remaining fold. This process is repeated k times, with each fold used once as the test set. The final score is the average of all k test result :
		CVerror​=(1/k)​∑(i=1,k)​errori
- Leave-One-Out Cross-Validation (LOO) : a special case of k-Fold where k = number of samples. Each observation is used once as the test set while the remaining data is used for training. It gives very accurate estimates but is computationally expensive for large datasets. 
- Hold out Method : the simplest technique where the dataset is split into two parts : a training set and a testing set (70% and 30%). The model is trained on the training set and evaluated on the test set.It is fast but may lead to biased results depending on the split.

14. __Difference Between Regularization , Standardization and normalization :__  
---
1. Regularization : A technique used to reduce overfitting by adding a penalty term to the model's loss function , discouraging overly complex models. Examples are L1 , L2 , Elastic Net. 
	   works on model parameters (weights) . 
2. Standardization : A preprocessing step that rescales features so they have mean = 0 and standard deviation = 1 : 
		​x' = (x−μ​)/σ <br>Useful for algorithms sensitive to feature scales like SVM , KNN , logistic regression etc 
	
	
3. Normalization : A preprocessing step that rescales feature values into a fixed range usually [0,1]  
		x' = (x - xmin) /(xmax - xmin)
		<br> useful when features have different scales or units <br>


|Aspect|Regularization|Standardization|Normalization|
|---|---|---|---|
|Purpose|Prevent overfitting|Rescale features (mean = 0, std = 1)|Rescale features to a range (e.g., [0,1])|
|Works On|Model weights|Input features|Input features|
|Main Idea|Add penalty to loss function|Center and scale features|Shrink features into fixed range|
|Example Techniques|L1, L2, Elastic Net|Z-score scaling|Min-Max scaling|
|When to Use|High variance / overfitting|Algorithms needing Gaussian-like distribution|Features with different ranges / uni|
15. what is Feature engineering in machine learning ; 
---
Is the process of creating , transforming or selecting relevant features from raw data to improve the performance of machine learning model. Better features often lead to better model accuracy and generalization. It also reduces overfitting and make the model easier to interpret.
__key steps in feature engineering:__
- __Feature creation__ : generate new features from existing data like extracting "year" or  "month" from a date column 
- __Feature transformation__ : apply scaling , normalization or mathematical transformations (log , square root) to features.
- __feature encoding__ : convert categorical variables into numerical form like one-hot encoding , label encoding. 
- __feature selection__ : identify and keep only the most relevant features using techniques like correlation analysis , mutual information ir model-based importance scores.
Examples : <br>
- raw data : Date of birth , feature engineered : Age . 
- raw data : text review -> feature engineered : sentiment score .

16. difference between feature engineering and feature selection : 
___

| Aspect     | feature engineering                                                                                     | feature selection                                                                                                        |
| ---------- | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Definition | Process of creating, transforming, or deriving new features from raw data to improve model performance. | Process of selecting the most relevant features from the existing dataset to reduce noise and improve model performance. |

|   |   |   |
|---|---|---|
|Purpose|To enhance or create meaningful features that the model can learn from.|To remove irrelevant or redundant features and simplify the model.|

|   |   |   |
|---|---|---|
|Process|Involves feature creation, transformation, encoding, scaling, etc.|Involves statistical tests, correlation analysis, mutual information, or model-based importance scores.|

|   |   |   |
|---|---|---|
|Output|New or transformed features added to the dataset.|Subset of the original features retained for modeling.|

|   |   |   |
|---|---|---|
|Example|Extracting Age from Date of Birth or generating sentiment scores from text.|Selecting top 10 features with highest importance from 50 features using Random Forest.|
17. Feature selection techniques in machine learning : 
---
F.S : is the process of choosing the most relevant features from your dataset to improve model performance , reduce overfitting and simplify the model .<br>
-  Filter methods : it evaluate each feature independetly with target variable . feature with high correlation with target variable are selected as it means this feature has some relation and can help us in making predictions . here features are selected based on statistical measures without involving any machine learning model . <br>
	examples :
	-  correlation coefficient : remove features highly correlated with others 
	- chi square test : for categorical features
	- anova F-test : for numerical features 
-  Wrapper methods : it uses different combination of features and compute relation between these subset features and target variable and based on conclustion addiction and removal of features are done. Stopping criteria for selecting the best subset are usually pre-defined by the person training the model such as when the performance of the model decreases or a specific number of features are achieved.
		examples : 
		 - forward selection : start with no features and add one at a time 
		 - backward elimination : start with all features and remove one at a time
		 - recursive feature elimination RFE: iteratively removes least important features using model weights
-  Embedded methods :it perform feature selection during the model training process allowing the model to select the most relevant features based on the training process dynamically 
		examples : 
		Lasso regression : can shrink some feature coefficients to zero 
		decision tree / random forest feature importance : select features based on importance socres learned during training .
<br>
17. what is Dimentionality reduction in machine learning : 
---
Dimentionality reduction is the process of reducing the number of features (variables) in a dataset while retaining the most of important information. It helps in simplifying models , improving performance , reducing overfitting and speedind up computation. Feature selection and engineering comes under this.
- reduces computational cost for high-dimensional datasets 
- helps visualize data in 2D or 3D space. 
- reduces overfitting by removing irrelevant or noisy features
<br>
Example : a dataset has 100 features. using PCA , it can be reduced to 10 principal components that capture 95% of the variance.

19. what is categorical data and how to handle it ? 
---
refers to features that represent discrete values or categories m rather than continuous numerical values. Examples include gender(Male , female) , color (Red , Blue , Green) or product type (Electronics , clothing) .
<br>types of categorical data : 
- nominal : where the order does no matter 
- ordinal : when the order matter 
Machine learning models require numerical imputs , so the categorcial data needs to be handelled using encoding "label encoding , One-Hot encoding , Binary encoding , target/mean encoding ".

20. what is the upsampling and the downsampling : 
---
Upsampling and downsampling are techniques used to handle imbalanced datasets where the number of samples in different classes is unequal . 
- Upsampling(oversampling): increases the number of samples in the minority class to balance the dataset. Techniques include : 
<br>Random Oversampling : duplicate random samples from the minority class . <br>
SMOTE (synthetic minority over-sampling technique) : generate sunthetic samples by interpolating between existing minority samples 

- Downsampling (undersampling) : reduces the number of samples in the majority class to balance the dataset , techniques include  :  <br>
Random undersampling : randomly remoce samples from the mojority class 
<br>
Cluster based undersampling : remove samples based on clustering to retain diversity 
<br>
***Example:** We have a dataset of 1000 positive samples, 100 negative samples.

- Upsampling create 900 additional negative samples.
- Downsampling reduce positive samples to 100.

21. Explain SMOTE method used to handle imblance 
---
SMOTE creates synthetic data points for minority classes using linear interpolation between existing samples.
- the model is trained on more diverse examples rather than duplicating existing points 
- it may introduce noise into the dataset, potentially affecting model performance if overused.

22. how to handle missing and duplicate values : 
---
Missing values are common in real world datasets and can affect model performance. Techniques to handle missing values : 
- remove rows or columns : drop rows with missing values using dropna() in pandas , drop column if most values are missing 
- Imputation : Mean / Median / mode , forward/backward fill , prediction-based imputation 
- flag missing values : create a new binary column to indicate whether a value was missing . <br>
Duplicate rows can lead to biased or misleading results , techniques to handle it : 
- identify duplicates 
- remove duplicates 
- keep the most relevant row

23. what are outliers and how to handle them ? 
---
Outliers are data points that differ significantly from other observations in the dataset. They can arise due to errors , variability in data or rare events .
- can skew statistics like mean and standard deviation 
- can mislead machine learning models , especially regression and distance-based algorithms
<br>
Detection methods : <br>
- Box plot / IQR method 
- Z-score method
- visualization : Scatter plots , histograms or violin plots 
<br>
Handling methods : 
- Remove outliers : delete extreme values if they are errors or irrelevant 
- Transform data : apply log , square root or other transformations to reduce skewness 
- cap/floor values : replace extreme values with upper/lower bounds(Winsorization)
- Use Robust Models : models like decision trees or random Forests are less sensitive to outliers 

24. Different hypothesis in machine learning ? 
---
In machine learning , a hypothesis is a function or model that maps input features to output predictions. Different hypotheses represent different types of models or assumptions about the data .
	1. Null Hypothesis (H0) : 
			- Assumes no effect or no relationship exists between features and target
			- often used in statistical testing to validate model assumptions 
			- Example : Feature X has no impact on predicting Y 
	2. Alternative Hypothesis (H1 or Ha) : 
			- Assumes there is a relationship or effect.
	3. Parametric hypotheses :
			- Assume the data follows a known distribution and have fixed parameters 
			- Example : linear regression assume linear relationship with parameters (weights)
	4. Non-Parametric Hypotheses : 
			- Make no assumptions about the underlying data distribution 
			- Examples : Decision Trees , K-nearest neighbors
	5. Machine learning hypothesis functions (hθ):
			- represent the model used to make predictions 
			- example :hθ​(x)=θ0​+θ1​x1​+θ2​x2​+⋯+θn​xn​
			- in supervised learning the goal is to find the hypothesis that minimizes error on the training data

25. What is Bias-Variance tradeoff ? 
---
the bias-variance tradeoff is a fundamental concept in machine learning that describes the tradeoff between two sources of error that affect model performance. 
	1. Bias : 
		- Error due to wrong assumptions in the learning algorithm 