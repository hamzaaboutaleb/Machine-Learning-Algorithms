### what is regression ? 

__Regression__ : a statistical method used to analyze the relationship between a dependent variable and one or more independent variables .  <br>
it works by finding a model like the line of best fit that define the most how the independent variables affect the dependent one . <br>

In machine Learning it refers to a supervised learning technique that predict a continuous numerical value based on the independent variables . <br>


| Height | Weight |
| ------ | ------ |
| 180    | 83     |
| 150    | 45     |
if we want to predict the weight based on the height , the height column is called a __Features__ and the Weight column is called the __Target__ .<br>

#### type of linear regression : 
--- 
__Linear regression__ is one of the simplest and most widely used statistical models. This assumes that there is a linear relationship between the independent and dependent variables. This means that the change in the dependent variable is proportional to the change in the independent variables. For example predicting the price of a house based on its size.<br>



__Multiple linear regression__ extends simple linear regression by using multiple independent variables to predict target variable. For example predicting the price of a house based on multiple features such as size, location, number of rooms, etc.<br>

__Polynomial Regression__ is used to model with non-linear relationships between the dependent variable and the independent variables. It adds polynomial terms to the linear regression model to capture more complex relationships. For example when we want to predict a non-linear trend like population growth over time we use polynomial regression.<br>

__Ridge & Lasso Regression__ are regularized versions of linear regression that help avoid overfitting by penalizing large coefficients. When there’s a risk of overfitting due to too many features we use these type of regression algorithms.<br>

__Support Vector Regression (SVR)__ is a type of regression algorithm that is based on the __Support Vector Machine (SVM)__ algorithm. SVM is a type of algorithm that is used for classification tasks but it can also be used for regression tasks. SVR works by finding a hyperplane that minimizes the sum of the squared residuals between the predicted and actual values.<br>

__Decision tree__ Uses a tree-like structure to make decisions where each branch of tree represents a decision and leaves represent outcomes. For example predicting customer behavior based on features like age, income, etc there we use decison tree regression.<br>

__Random Forest__ is a ensemble method that builds multiple decision trees and each tree is trained on a different subset of the training data. The final prediction is made by averaging the predictions of all of the trees. For example customer churn or sales data using this.<br>

#### Regression Evaluation Metrics

Evaluation in machine learning measures the performance of a model. Here are some popular evaluation metrics for regression:<br>

- Mean Absolute Error (MAE):The average absolute difference between the predicted and actual values of the target variable.<br>
- Mean Squared Error (MSE) The average squared difference between the predicted and actual values of the target variable.<br>
- Root Mean Squared Error (RMSE) Square root of the mean squared error.<br>
- Huber Loss : A hybrid loss function that transitions from MAE to MSE for larger errors, providing balance between robustness and MSE’s sensitivity to outliers.<br>
- R2 – Score: Higher values indicate better fit ranging from 0 to 1.<br>

### Real world application of regression : 


__Predicting prices__ : Used to predict the price of a house based on its size, location and other features <br>
__Forecasting trends__ : Model to forecast the sales of a product based on historical data<br>
__Identifying risk facors__ : used to identify risk factors for heart patient based on medical records <br>
Making decisions : it could be used to recommend which stock to buy based on market data <br>
Businesses frequently use linear regression to comprehend the connection between advertising spending and revenue. For instance, they might apply the linear regression model using advertising spend as an independent variable or predictor variable and revenue as dependent variable , the equation would take the following form : revenue = Beta0 + Beta1 (ad spending) <br> it is used also in the medical field to understand the relation between drug dosage and blood pressure. <br>
agriculture scientists frequently use lr to see the impact of rainfall and fertilizer on the amount of fruits yielded. 
For instance, scientists might use different amounts of fertilizer and see the effect of rain on different fields and to ascertain how it affects crop yield. <br>
#### advantages of regression : 
- easy to understand and to interpret .
- Robust to outliers . 
- Can handle both linear relationships easy

#### Disadvantages of Regression : 
- Assumes linearity 
- sensitive to situation where two or more independent variables are highly correlated to each other i.e collinearity
- may not be suitable to highly complex relationships

### Simple linear regression :
---
### what kind of relationship can linear regression show ? 
---
- Positive relationship :
when the regression line between the two variables moves in the same direction with an upward slope , the variables are said to be in a positive relationship. if x is increased -> there will be an increase in the dependent variable.
- Negative relationship : 
when the regression line between the two variables move in the same direction as the downward slope , the variables are said to be in a negative relationship . if one is increased -> the other dicreased. 
- No relationship : 
when the bestfit line is flat  , there will be no change in the dependent variable by increasing or dicreasing the independents variables .<br> 
You can see the type of relationship using correlation or the covariance .<br>
__Note__ : Covariance shows the direction of the relationship but it doesnt say how positive or negative the relationship is . to know this remember that if the covariance value is negative and if the dependent variable increases , the depencdent variable decreases and vice versa. <br>
Correlation is a statistical measure that shows the direction of the relationship as well as the strenght of the relationship. the range of correlation is between -1 and +1 . its called a perfect correlation if all points fall on the best fit line - which is very __unlikely__ <br>
#### Least square method : 
---
the main idea of linear regression model is to ful a line that is the best fit for the data .For this , you use a technique called __least square method__. In layman's terms, it is the process of fitting the best curve for a set of data points by reducing the distance between the actual value and predicted value (sum of squared residuals) . the distance between both values is often known as error or variation or variance . <br> 
its known that the equation of a straight line is y = ax + b<br> similarly , the equation of the best line for linear regression is : 
![[images/659809bf531ac2845a27252e_image21_11zon_b29a6e7ebf.avif]]
y = B0 + x1B1<br>
Meanwhile, since there are more than 2 independent variables in multiple linear regression the equation become : 
y = B0 + x1B1 + ... + xnBn (x is called the explanatory variable)

### how you do linear regression ? 
--- 
By an example , lets say you want to know to what degree the tip amount can be predicted by the bill studied. the tip is the dependent variable , the bill is the independent variable . <br>
to fit the best fit line you need to minimize the sum of __squared errors__
- Step 1 : check if there is a linear relationship between the variables.
u already know that the equation of a line , make a scatter plot to see a relationship between the variables. *Remember that the best fit line will always pass though the centroid* .
- Step 2 : check the correlation of the data : 
 After plotting a scatter plot and knowing what type of relationship it has, calculate the correlation to know the direction’s strength. In this case, the correlation is 0.866, which shows that the relationship is very strong.
 - step 3 : the calculations : 
 The equation of best-fit line is: Ŷ = x*β1+β0  
where β1 is the coefficient of regression or slope. To predict Ŷ, you need to know this coefficient. It will also tell you the change in dependent variable if you increase the independent variable by 1 unit. The formula for finding this is: <br> ![[images/659809c7531ac2845a272536_image4_11zon_d6c6352c9b.avif]]
and the constant term is calculated by B0 = y_bar - y_bar + B1 <br>
a note : these two together give you the **closed-form (analytical)** solution for the line of best fit — used in **simple linear regression (one feature)**.
That means:

- You can compute the exact slope (β1\beta_1β1​) and intercept (β0\beta_0β0​) directly from the data.
    
- No iterations or optimization steps are required. <br>
 _So why do we need gradient descent?_
---
Because the formula above only works **when you have a simple linear regression with one or a few features** — and when the relationship is **linear**.

When the problem becomes more complex, that formula becomes **impossible or extremely expensive** to compute. <br>
__1. For Multiple Linear Regression__

If you have many variables (x₁, x₂, …, xₙ), the closed-form solution becomes:

**β = (XᵀX)⁻¹ Xᵀy**

But this requires computing a **matrix inverse**, which:

- Is very expensive when the number of features is large.
- Can even be impossible if (XᵀX) is **not invertible (singular matrix)**.

__ 2. For Nonlinear Models or Neural Networks__

If your model is not linear — e.g.,  
**y = f(x; θ)**

where *f* is a neural network —  
there is **no closed-form formula** for the optimal parameters θ.  
You can’t compute derivatives analytically like in the linear case.

In those cases, **gradient descent** helps us find approximate optimal parameters by minimizing a cost function (like MSE).

### Coefficient of determination (R2) : 
___
The **coefficient of determination**, denoted as R^2, measures **how well the regression line fits the data**.<br>
it tells you what propostion of the variance in the dependent variable is explained by the independent variable(s).
![[Pasted image 20251113200435.png]]
its used to know the accuracy of the model . 

**Formula**

R² = 1 - (SS_res / SS_tot)

Where:  
- SS_res = Σ (y_i - ŷ_i)² → **Residual Sum of Squares** (model error)  
- SS_tot = Σ (y_i - ȳ)² → **Total Sum of Squares** (total variation in data)

---

### **Intuitive Explanation**

R² answers the question:  
> “How well does my line explain the data compared to just using the average value?”

- If all data points lie exactly on the line → **R² = 1** (perfect fit)  
- If the line doesn’t explain anything → **R² = 0**  
- If the line is worse than guessing the mean → **R² < 0**

---

### **Step-by-Step Example**

| House | Actual (y) | Predicted (ŷ) |
|:--|:--|:--|
| 1 | 200 | 210 |
| 2 | 250 | 240 |
| 3 | 300 | 310 |

1. Compute the mean of y:  

ȳ = (200 + 250 + 300) / 3 = 250

2. Compute total variation:  

SS_tot = (200 - 250)² + (250 - 250)² + (300 - 250)² = 5000

3. Compute model error:  

SS_res = (200 - 210)² + (250 - 240)² + (300 - 310)² = 300

4. Compute R²:  

R² = 1 - (300 / 5000) = 0.94

**Interpretation:**  
The model explains **94% of the variance** in house prices — a very good fit.

###  Adjusted R-squared
Every time you add a new input variable, there will be an increase in the R square. Hence, it’s not wise to use the R square to decide whether to add a new input variable. To address this, a quantity known as "adjusted R-squared", which is a modified version of R-squared, is used. It’s more useful when you add irrelevant variables to the model. If you add variables that do not affect the target variable, the adjusted R-squared value will decrease and R-squared value will increase. Note that it is always lower than the R square.

Usually, the value of R-squared and adjusted R-squared is somewhat the same. But, if you see a large difference, you need to check your independent variables again and see if there is any relationship between the target variable and the independent variable.