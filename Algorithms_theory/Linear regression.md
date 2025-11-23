## what linear regression is really doing ? 
---
- Linear regression finds the best linear explanation of how inputs produce outputs . i.e linear regression finds the vector of parameters that gives predictions closest(in euclidean distance) to the real output 
#### the true core idea :
---
Imagine you have : 
- A set of inputs (features)
- An output you want to explain 
Linear regression assumes : the output is a combination of the inputs plus some noise . 
### idea 3 : 
---
Linear regression tries to find the relationship , not just a line.
- How much does the output change when the input changes ? 
- what is the best possible straight line rule that explains the data ? 
- which linear combination of inputs gets me closest to reality ? 

### The hidden Geometric intuition 
---
think of : 
- All your features as forming vectors in a high dimensional space 
- Your target variable as another vector in the sampe space
Linear regression tries to answer the questions : 
- what is the projection of the ttarget vector onto the space spanned by feature vectors 

### why squared error ? 
---
- squared error measures Euclidean distance 
- so minimizing the squared error = minimizing distance between : 
	- actual outputs vector
	- predicted outputs vector

## Deriving linear regression mathematically 
---
- **Write the model**
    
- **Write the loss function**
    
- **Rewrite using matrix notation**
    
- **Take the derivative and set to zero**
    
- **Solve the normal equation**
    
- **Interpret geometrically**

### 1 - Write the model : 
---
Assume you have nnn data points and ppp features.

The linear regression model is:

							y=Xβ+ε
Where:

- y is an n x 1 vector of outputs 
- X is an nxp vector of inputs 
- β is a px1 vector of parameters 
- epsilon is noise
we want to find the best β

### 2 - define the loss (sum of squared errors ) : 
---
Predictions: 
			y^​=Xβ
Residuals (erros) : 
			e=y−Xβ