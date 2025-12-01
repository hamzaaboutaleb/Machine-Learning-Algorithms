## Introduction to linear regression : 
---
### 1 . what is regression : 
---
a statistical method used to find and model the relationship between variables , it helps predict or understand how changes in the independent variable are associated with the changes of the dependent variable.<br>
__Linear Regression__ is the most common form of this technique. It establishes the linear relationship between variables.

### 2 . Difference between Regression and Classification : 
--- 
Regression predicts a continuous numerical value (e.g. Price , temperature), while classification rpedicts a discrete category (e.g , a spam or not spam) <br>
the main difference is the target variable, regression uses a continuous one and classification uses a categorical one

### 3. Real world application : 
---
- Businesses frequently use linear regression to comprehend the connection between advertising spending and revenue. For instance, they might apply the linear regression model using advertising spend as an independent variable or predictor variable and revenue as the response variable.
- Linear regression may be used in the medical field to understand the relationships between drug dosage and patient blood pressure. Researchers may manage different measurements of a specific medication to patients and see how their circulatory strain reacts/blood pressure responds. They might fit a model using dosage as an independent variable and blood pressure as the dependent variable.
- Agriculture scientists frequently use linear regression to see the impact of rainfall and fertilizer on the amount of fruits/vegetables yielded. For instance, scientists might use different amounts of fertilizer and see the effect of rain on different fields and to ascertain how it affects crop yield. They might fit a multiple linear regression using rainfall and fertilizer as the predictor variables and crop yield as the dependent variable or response variable

## The straight line : 
---
### 1 . the equation of the line : 
---
- In linear regression, the best-fit line is the straight line that most accurately represents the relationship between the independent variable (input) and the dependent variable (output). It is the line that minimizes the difference between the actual data points and the predicted values from the model.
- the goal of linear regression is to find a straight line that minimizes the error (the difference) between the observed data points and the predicted values. This line helps us predict the dependent variable for new, unseen data.
- For simple linear regression (with one independent variable), the best-fit line is represented by the equation  __y=mx+b__.
- The best-fit line will be the one that optimizes the values of m (slope) and b (intercept) so that the predicted y values are as close as possible to the actual data points.

### 2. Minimizing the error : the least squares method
---
to find the best fit line we use a method called Least Squares.The idea behind this method is to minimize the sum of squared differences between the actual values (data points) and the predicted values from the line. These differences are called residuals. <br>
the formula for residuals is : yi - y_hat_i
<br>
The least squares method minimizes the sum of the squared residuals.This method ensures that the line best represents the data where the sum of the squared differences between the predicted values and actual values is as small as possible.

### 3. Interpretation of the best-Fit line : 
---
- Slope(m) : The slope of the best-fit line indicates how much the dependent variable (y) changes with each unit change in the independent variable (x). For example if the slope is 5, it means that for every 1-unit increase in x, the value of y increases by 5 units.
- Intercept(b) : The intercept represents the predicted value of y when x = 0. It’s the point where the line crosses the y-axis.
### 4. Limitations : 
---
- Assumes linearity : this method assumes the relationship between the variables is linear. If the relationship is non-linear, linear regression might not work well.
- Sensitivity to outliers : Outliers can significantly affect the slope and intercept, skewing the best-fit line.

## Model Definition : 
---
### 1 . Hypothesis function : 
---
In linear regression, the hypothesis function is the equation used to make predictions about the dependent variable based on the independent variables. It represents the relationship between the input features and the target output.<br>
For a simple case with one independent variable, the hypothesis function is:
h(x)=β₀+β₁x <br>
for multiple linear regression :  h(x₁,x₂,...,xₖ)=β₀+β₁x₁+β₂x₂+...+βₖxₖ

### 2 . Assumptions of the linear regression : 
---
- Linearity :the relationship between inputs (X) and the output (Y) is a straight line.
- Independence of errors : The errors in predictions should not affect each other.
- Constant variance ****(Homoscedasticity):**** The errors should have equal spread across all values of the input. If the spread changes (like fans out or shrinks), it's called heteroscedasticity and it's a problem for the model.
- Normality of errors :  the errors should follow a normal distribution 
- No multicollinearity : input variables shouldnt be too closely related to each other 
- No autocorrelation : errors shouldnt show repeating patterns especially in time based data
- additivity : the total effect on Y is just the sum of effects from each X , no mixing or interaction between them 

### 3. type of linear regression : 
---
- simple linear regression : is used when we want to predict a target value using only one input feature. It assumes a straight line relationship between the two. 
- Multiple linear regression involves more than one independent variable and one dependent variable.

### 4 . Cost function : 
--- 
In Linear Regression, the cost function measures how far the predicted values (Y^) are from the actual values (Y). It helps identify and reduce errors to find the best-fit line. The most common cost function used is Mean Squared Error (MSE), which calculates the average of squared differences between actual and predicted values. <br>To minimize this cost, we use Gradient Descent, which iteratively updates θ1 and θ2​ until the MSE reaches its lowest value. This ensures the line fits the data as accurately as possible.

### 5. Gradient descent for linear regression : 
---
Gradient descent is an optimization technique used to train a linear regression model by minimizing the prediction error. It works by starting with random model parameters and repeatedly adjusting them to reduce the difference between predicted and actual values.<br>
How it works:

- Start with random values for slope and intercept.
- Calculate the error between predicted and actual values.
- Find how much each parameter contributes to the error (gradient).
- Update the parameters in the direction that reduces the error.
- Repeat until the error is as small as possible.

### 6. evaluation metrics for linear regression : 
---
A variety of evaluation measures can be used to determine the strength of any linear regression model. These assessment metrics often give an indication of how well the model is producing the observed outputs. the commont measurements are : 
- Mean square error : is an evaluation metric that calculates the average of the squared differences between the actual and predicted values for all the data points. The difference is squared to ensure that negative and positive differences don't cancel each other out.MSE is a way to quantify the accuracy of a model's predictions. MSE is sensitive to outliers as large errors contribute significantly to the overall score.
- Mean absolute error :is an evaluation metric used to calculate the accuracy of a regression model. MAE measures the average absolute difference between the predicted values and actual values.Lower MAE value indicates better model performance. It is not sensitive to the outliers as we consider absolute differences.
- The square root of the residuals' variance is the Root mean squared error . It describes how well the observed data points match the expected values or the model's absolute fit to the data. In mathematical notation.RMSE is in the same unit as the target variable and highlights larger errors more clearly.
- Coefficient of Determination (R-squared): is a statistic that indicates how much variation the developed model can explain or capture. It is always in the range of 0 to 1. In general, the better the model matches the data, the greater the R-squared number.  
- Adjusted R2R2measures the proportion of variance in the dependent variable that is explained by independent variables in a regression model. Adjusted R square accounts the number of predictors in the model and penalizes the model for including irrelevant predictors that don't contribute significantly to explain the variance in the dependent variables.

## Regularization techniques for linear models : 
---
### Lasso regression : (L1)
---
Lasso regression is a technique used for regularizing a linear regression model, it adds a penalty term to the linear regression objective function to prevent overfitting. 

### Ridge Regression : (l2)
--- 
is a linear regression technique that adds a regularization term to the standard linear objective. Again, the goal is to prevent overfitting by penalizing large coefficient in linear regression equation. It useful when the dataset has multicollinearity where predictor variables are highly correlated.
### Elastic Net Regression
---
s a hybrid regularization technique that combines the power of both L1 and L2 regularization in linear regression objective.