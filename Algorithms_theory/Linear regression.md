## Regression : 
---
### Definition : 
---
Regression is a statistical method used to analyze the relationship between dependent variable and one or more independent variables.<br>
it works by finding a model (like best fit line) that define how the dependent variable can be affected by the independent ones .<br>In machine learning it refers to a supervised machine learning algorithm that predict a continuous dependent variable based on independent ones.

### Types :
---
__Linear regression__ is one of the simplest and most widely used statistical models. This assumes that there is a linear relationship between the independent and dependent variables. This means that the change in the dependent variable is proportional to the change in the independent variables. For example predicting the price of a house based on its size.<br>
__Multiple linear regression__ extends simple linear regression by using multiple independent variables to predict target variable. For example predicting the price of a house based on multiple features such as size, location, number of rooms, etc.<br>
__Polynomial Regression__ is used to model with non-linear relationships between the dependent variable and the independent variables. It adds polynomial terms to the linear regression model to capture more complex relationships. For example when we want to predict a non-linear trend like population growth over time we use polynomial regression.<br>
__Ridge & Lasso Regression__ are regularized versions of linear regression that help avoid overfitting by penalizing large coefficients. When there’s a risk of overfitting due to too many features we use these type of regression algorithms.<br>
__Support Vector Regression (SVR)__ is a type of regression algorithm that is based on the __Support Vector Machine (SVM)__ algorithm. SVM is a type of algorithm that is used for classification tasks but it can also be used for regression tasks. SVR works by finding a hyperplane that minimizes the sum of the squared residuals between the predicted and actual values.<br>
__Decision tree__ Uses a tree-like structure to make decisions where each branch of tree represents a decision and leaves represent outcomes. For example predicting customer behavior based on features like age, income, etc there we use decison tree regression.<br>
__Random Forest__ is a ensemble method that builds multiple decision trees and each tree is trained on a different subset of the training data. The final prediction is made by averaging the predictions of all of the trees. For example customer churn or sales data using this.

### HOW  ? 
--- 
1. Normal Equation : 
---
The normal equation is a mathematical formula that provides a straightforward way to calculate the coefficients (β\betaβ) in linear regression. Instead of using trial-and-error or iterative methods, the normal equation allows us to find the best coefficients directly. The formula for the normal equation is:

### Real world application of regression : 
---


__Predicting prices__ : Used to predict the price of a house based on its size, location and other features <br>
__Forecasting trends__ : Model to forecast the sales of a product based on historical data<br>
__Identifying risk facors__ : used to identify risk factors for heart patient based on medical records <br>
Making decisions : it could be used to recommend which stock to buy based on market data <br>
Businesses frequently use linear regression to comprehend the connection between advertising spending and revenue. For instance, they might apply the linear regression model using advertising spend as an independent variable or predictor variable and revenue as dependent variable , the equation would take the following form : revenue = Beta0 + Beta1 (ad spending) <br> it is used also in the medical field to understand the relation between drug dosage and blood pressure. <br>
agriculture scientists frequently use lr to see the impact of rainfall and fertilizer on the amount of fruits yielded. 
For instance, scientists might use different amounts of fertilizer and see the effect of rain on different fields and to ascertain how it affects crop yield. <br>

## Mathematical foundations : 
---

Linear Regression models the relationship between input x and output y using:

$y = \beta_0 + \beta_1 x + \varepsilon$

**Where:**

- Beta0 : intercept
    
- Beta1: slope
    
- Epsilon: error term <br>
__Multiple features__
<br>
$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n + \varepsilon$

__vector form__
$\hat{y} = X\beta$
