### what is regression ? 

__Regression__ : a statistical method used to analyze the relationship between a dependent variable and one or more independent variables .  <br>
it works by finding a model like the line of best fit that define the most how the independent variables affect the dependent one . <br>

In machine Learning it refers to a supervised learning technique that predict a continuous numerical value based on the independent variables . <br>


| Height | Weight |
| ------ | ------ |
| 180    | 83     |
| 150    | 45     |
if we want to predict the weight based on the height , the height column is called a __Features__ and the Weight column is called the __Target__ .<br>

##### type of linear regression : 
--- 
__Linear regression__ is one of the simplest and most widely used statistical models. This assumes that there is a linear relationship between the independent and dependent variables. This means that the change in the dependent variable is proportional to the change in the independent variables. For example predicting the price of a house based on its size.<br>



__Multiple linear regression__ extends simple linear regression by using multiple independent variables to predict target variable. For example predicting the price of a house based on multiple features such as size, location, number of rooms, etc.<br>

__Polynomial Regression__ is used to model with non-linear relationships between the dependent variable and the independent variables. It adds polynomial terms to the linear regression model to capture more complex relationships. For example when we want to predict a non-linear trend like population growth over time we use polynomial regression.<br>

__Ridge & Lasso Regression__ are regularized versions of linear regression that help avoid overfitting by penalizing large coefficients. When there’s a risk of overfitting due to too many features we use these type of regression algorithms.<br>

__Support Vector Regression (SVR)__ is a type of regression algorithm that is based on the __Support Vector Machine (SVM)__ algorithm. SVM is a type of algorithm that is used for classification tasks but it can also be used for regression tasks. SVR works by finding a hyperplane that minimizes the sum of the squared residuals between the predicted and actual values.<br>

__Decision tree__ Uses a tree-like structure to make decisions where each branch of tree represents a decision and leaves represent outcomes. For example predicting customer behavior based on features like age, income, etc there we use decison tree regression.<br>

__Random Forest__ is a ensemble method that builds multiple decision trees and each tree is trained on a different subset of the training data. The final prediction is made by averaging the predictions of all of the trees. For example customer churn or sales data using this.<br>

## Regression Evaluation Metrics

Evaluation in machine learning measures the performance of a model. Here are some popular evaluation metrics for regression:<br>

- Mean Absolute Error (MAE):The average absolute difference between the predicted and actual values of the target variable.<br>
- Mean Squared Error (MSE) The average squared difference between the predicted and actual values of the target variable.<br>
- Root Mean Squared Error (RMSE) Square root of the mean squared error.<br>
- Huber Loss : A hybrid loss function that transitions from MAE to MSE for larger errors, providing balance between robustness and MSE’s sensitivity to outliers.<br>
- R2 – Score: Higher values indicate better fit ranging from 0 to 1.<br>

