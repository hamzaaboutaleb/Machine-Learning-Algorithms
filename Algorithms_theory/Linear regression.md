## Introduction to linear regression : 
---
Linear regression is a type of supervised machine learning algorithm that learns from the labelled datasets and maps the data points with most optimized linear functions which can be used for prediction on new datasets. It assumes that there is a linear relationship between the input and output, meaning the output changes at a constant rate as the input changes. This relationship is represented by a straight line.<br>
we use independent variable(s) to predict dependent variable.
### best fit line in linear regression : 
in linear regression , the best fit line is the straight line that most accurately represents the relationship between the independent variable (input) and the dependent variable (output). It is the line that minimizes the difference between the actual data points and the predicted values from the model.
1. __Goal of the Best fit line__ : 
---
The goal of linear regression is to find a straight line that minimizes the error (the difference) between the observed data points and the predicted values. This line helps us predict the dependent variable for new, unseen data.

2. __Equation of the best fit line__ 
--- 
For simple linear regression (with one independent variable), the best-fit line is represented by the equation :  __y=mx+b__ 

3. __Minimizing the error__
---
to find the best fir line  we use a method called Least Squares . The idea behind this method is to minimize the sum of squared differences between the actual values (data points) and the predicted values from the line. These differences are called residuals.
Residual = yi - y_hat i 

4. __interpretation of the Best fit line__
---
- Slope (m) : The slope of the best-fit line indicates how much the dependent variable (y) changes with each unit change in the independent variable (x). For example if the slope is 5, it means that for every 1-unit increase in x, the value of y increases by 5 units.
- **Intercept (b):*** The intercept represents the predicted value of y when x = 0. It’s the point where the line crosses the y-axis.
in linear regression some hypothesis are made to ensure reliability of the model's results : 
- assumes linearity : the method assumes the relationship between the variables is linear 
- sensitivity to outliers : outliers can significantly affect the slope and intercept , skewing the best fit line 
<br> __Hypothesis function in linear regression__ : 
---
in linear regression , the hypothesis function is the equation used to make predictions about the dependent variable based on the independent variables. It represents the relationship between the input features and the target output .

###  Cost function for Linear Regression : 
---
In Linear Regression, the cost function measures how far the predicted values (Y^) are from the actual values (Y). It helps identify and reduce errors to find the best-fit line. The most common cost function used is Mean Squared Error (MSE), which calculates the average of squared differences between actual and predicted values. <br>
To minimize this cost, we use Gradient Descent, which iteratively updates θ1 and θ2​ until the MSE reaches its lowest value. This ensures the line fits the data as accurately as possible.

### Gradient descent for linear regression :
---

Gradient descent is an optimization technique used to train a linear regression model by minimizing the prediction error. It works by starting with random model parameters and repeatedly adjusting them to reduce the difference between predicted and actual values.
How it works:

- Start with random values for slope and intercept.
- Calculate the error between predicted and actual values.
- Find how much each parameter contributes to the error (gradient).
- Update the parameters in the direction that reduces the error.
- Repeat until the error is as small as possible.

### Evaluation metrics for linear regression : 
---
A variety of evaluation measures can be used to determine the strength of any linear regression model. These assessment metrics often give an indication of how well the model is producing the observed outputs.

The most common measurements are: 
1. Mean square error 
2. mean absolute error 
3. root mean squared error 
4. coefficient of determination 
5. adjusted r squared error 

# Regularization techniques for linear models : 
---
