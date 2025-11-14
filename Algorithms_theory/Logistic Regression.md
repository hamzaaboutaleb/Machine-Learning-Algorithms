### Definition : 
A supervised machine learning algorithm used for classification problem , unlike linear regression that predict continuous values it predict the probability that an input belongs to a specific class. It is used for binary classification where the output can be one of two possible categories (like true/false). <br>
### type of logistic regression : 
Logistic regression can be classified into three main types based on the nature of the dependent variable:

1. *Binomial Logistic Regression*: This type is used when the dependent variable has only two possible categories. Examples include Yes/No, Pass/Fail or 0/1. It is the most common form of logistic regression and is used for binary classification problems.
2. *Multinomial Logistic Regression*: This is used when the dependent variable has three or more possible categories that are not ordered. For example, classifying animals into categories like "cat," "dog" or "sheep." It extends the binary logistic regression to handle multiple classes.
3. *Ordinal Logistic Regression*: This type applies when the dependent variable has three or more categories with a natural order or ranking. Examples include ratings like "low," "medium" and "high." It takes the order of the categories into account when modeling.
### Assumptions of Logistic Regression 

Understanding the assumptions behind logistic regression is important to ensure the model is applied correctly, main assumptions are:

1. ****Independent observations****: Each data point is assumed to be independent of the others means there should be no correlation or dependence between the input samples.
2. ****Binary dependent variables****: It takes the assumption that the dependent variable must be binary, means it can take only two values. For more than two categories Softmax functions are used.
3. ****Linearity relationship between independent variables and log odds****: The model assumes a linear relationship between the independent variables and the log odds of the dependent variable which means the predictors affect the log odds in a linear way.
4. ****No outliers****: The dataset should not contain extreme outliers as they can distort the estimation of the logistic regression coefficients.
5. ****Large sample size****: It requires a sufficiently large sample size to produce reliable and stable results.

### Understanding the softmax function 

- it is an important step in the logistic regression which is used to convert the raw output of the model into a probabilty value between 0 and 1 .
-  This function takes any real number and maps it into the range 0 to 1 forming an "S" shaped curve called the sigmoid curve or logistic curve. Because probabilities must lie between 0 and 1, the sigmoid function is perfect for this purpose.
- In logistic regression, we use a threshold value usually 0.5 to decide the class label:
	- If the sigmoid output is same or above the threshold, the input is classified as Class 1.
	- If it is below the threshold, the input is classified as Class 0.
<br>
This approach helps to transform continuous input values into  meaningful class predictions.

### How does logistic regression work? 
