PCA is a dimensionality reduction technique and helps us to reduce  the number of features in a dataset while keeping the most important information. It changes complex dataset by transforming correlated features into a smaller set of uncorrelated __components__ .
<br> 
it helps us to remove redundancy , inprove computational efficiency and make data easier to visualize and analyze.
## How PCA works : 
---
PCA uses linear algebra to transform data into new features called principal components. It finds these by calculating eigenvectors(directions) and eigenvalues (importance) from the covariance matrix . PCA selects the top components with the highest eigenvalues and projects the data onto them simplify the dataset
<br> Note : __It prioritizes the directions where the data varies the most because more variation = more useful information.__
<br>
Imagine you're looking at a messy cloud of data points like stars in the sky and want to simplify it. PCA helps you find the "most important angles" to view this cloud so you don’t miss the big patterns. Here’s how it works step by step: 
### Step 1 : Standarize the Data 
---
Different features may have different units and scales like salary vs age. To compate them fairly PCA first standardizes the data by making each feature have : 
- A mean of 0 
- A standard deviation of 1 

### Step 2: Calculate the covariance matrix 
---
Next PCA calculates the covariance matrix to see how features relate to each other whether they increase or decrease together. the calue of covariance can be positive negative or zeros.

### Step 3 : find the principal components : 
---
PCA identifies new axes where the data spreads out the most : 
- 1st Principal component : the direction of maximum variance 
- 2nd principal component : the next best direction , perpendiculat to pc1 and so on .
These directions come from the eigenvectors of the covariance matrix and their importance is measured by eigenvalues. For a square matrix A an eigenvector X (a non-zero vector) and its corresponding eigenvalue λ satisfy:

> AX=λX

this means : 
- When __A__ acts on X it only stretches or shrinks X by the scalar λ.
- The direction of X remains unchanged hence eigenvectors define "stable directions" of A.
Eigenvalues help rank these directions by importance.

### step 4 : Pick the top directions & transform data
---
After calculating the eigenvalues and eigenvectors PCA ranks them by the amount of information they capture. We then : 
1. select the top K components that capture most of the variance like 95%
2. transform the original dataset by projecting it onto these top components
This means we reduce the number of features (dimensions) while keeping the important patterns in the data 

## Advantages of PCA 
---
1. ***Multicollinearity Handling:** Creates new, uncorrelated variables to address issues when original features are highly correlated.
2. **Noise Reduction:*** Eliminates components with low variance enhance data clarity.
3. ***Data Compression:** Represents data with fewer components reduce storage needs and speeding up processing.
4. **Outlier Detection:*** Identifies unusual data points by showing which ones deviate significantly in the reduced space.

## Disadvantages of PCA :
---
1. ***Interpretation Challenges:*** The new components are combinations of original variables which can be hard to explain.
2. **Data Scaling Sensitivity:*** Requires proper scaling of data before application or results may be misleading.
3. ***Information Loss:** Reducing dimensions may lose some important information if too few components are kept.
4. **Assumption of Linearity:** Works best when relationships between variables are linear and may struggle with non-linear data.
5. **Computational Complexity:*** Can be slow and resource-intensive on very large datasets.
6. **Risk of Overfitting:** Using too many components or working with a small dataset might lead to models that don't generalize well.