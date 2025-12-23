### **What are the differences between supervised and unsupervised learning?**
|Supervised Learning|Unsupervised Learning|
|---|---|
|- Uses known and labeled data as input<br>- Supervised learning has a feedback mechanism <br>- The most commonly used supervised learning algorithms are decision trees, logistic regression, and support vector machine|- Uses unlabeled data as input<br>- Unsupervised learning has no feedback mechanism <br>- The most commonly used unsupervised learning algorithms are k-means clustering, hierarchical clustering, and apriori algorithm|
### 2. How is logistic regression done?
Logistic regression measures the relationship between the dependent variable (our label of what we want to predict) and one or more independent variables (our features) by estimating probability using its underlying logistic function (sigmoid).
### 3. Explain the steps in making a decision tree

1. Take the entire data set as input
2. Calculate entropy of the target variable, as well as the predictor attributes
3. Calculate your information gain of all attributes (we gain information on sorting different objects from each other)
4. Choose the attribute with the highest information gain as the root node 
5. Repeat the same procedure on every branch until the decision node of each branch is finalized
### **4. How do you build a random forest model?**
A random forest is built up of several decision trees . if you split thbre data into different packages and make a decision tree in each of the different data groups, the random forest brings all those trees together. <br>
Steps : 
1. Randomly select K features from a total of m features where k << m 
2. amonh the k features, calculate node D using the best split point

### **5. How can you avoid overfitting your model?**
overfitting refers to when the model that is only set for a very small amount of data and ignores the bigger picture. There are 3 main methods to avoid overfitting: 
1. keep the model simple - take fewer variables into account , thereby removing some of the noise in the training data 
2. use cross validation techniques such as k folds cross validation 
3. use regularization techniques such as LASSP that penalize certain model parameters if theyre likely to cause overfitting
### **6. Differentiate between univariate, bivariate, and multivariate analysis.**

Univariate : the univariate data contains only one variable . The purpose of univariate analysis is to describe the data and find patterns within it. Example : Height of students <br>
Bivariate : Bivariate data involves two different variables. The analysis of this type of data deals with causes and relationships and the analysis is done to determine the relationship between the two variables. <br>

Multivariate : Multivariate data involves three or more variables, it is categorized under multivariate. It is similar to a bivariate but contains more than one dependent variable.

### **7. What feature selection methods are used to select the right variables?**
There are two main methods for feature selection, i.e., filter and wrapper methods.
1.  **Filter Methods**

This involves: 

- Linear discrimination analysis
- [ANOVA](https://www.simplilearn.com/tutorials/statistics-tutorial/what-is-annova-test "ANOVA")
- [Chi-Square](https://www.simplilearn.com/tutorials/statistics-tutorial/chi-square-test "Chi-Square")

The best analogy for selecting features is "bad data in, bad answer out." When we limit or select features, we're all about cleaning up the data coming in.
2.  **Wrapper Methods**

This involves: 

- Forward Selection: We test one feature at a time and keep adding them until we get a good fit
- Backward Selection: We test all the features and start removing them to see what works better
- Recursive Feature Elimination: Recursively looks through all the different features and how they pair together

Wrapper methods are very labor-intensive, and high-end computers are needed if a lot of data analysis is performed with the wrapper method.

### You are given a data set consisting of variables with more than 30 percent missing values. How will you deal with them?
The following are ways to handle missing data values:

If the data set is large, we can just simply remove the rows with missing data values. It is the quickest way; we use the rest of the data to predict the values.

For smaller data sets, we can substitute missing values with the mean or average of the rest of the data using the pandas' data frame in python. There are different ways to do so, such as df.mean(), df.fillna(mean).

### what are dimensionality reduction and its benefits ? 
---
refers to the process of converting a data set with vast dimensions into data with fewer dimensions (fields) to convey similar information concisely.
the reduction helps in compressing data and reducing storage space. It also reduces computation time as fewer dimensions lead to less computing. It removes redundant features. (there s no need to store the same value in two different unit meters and inches )

### how should you maintain a deployed model ? 
---
The steps to maintain a deployed model are : 
- Monitor : constant monitoring of all models is needed to determine their performance accuracy. when you change something , you want to figure out how your changes are going to affect things. This needs to be monitored to ensure its doing what its supposed to do.
- Evaluate : evaluation metrics of the current model are calculated to determine if a new algorithm is needed.
- compare : the new models are compared to each other to determine which model performs the best 
- Rebuild : the best performing model is rebuilt on the current state of data.

### what are recommender systems ? 
---
A recommender system predicts how a user would rate a specific product based on their preferences. It can be split into two different areas : 
- collaborative filtering : As an example, Last.fm recommends tracks that other users with similar interests play often. This is also commonly seen on Amazon after making a purchase; customers may notice the following message accompanied by product recommendations: "Users who bought this also bought…"
- content based filtering : For example, Pandora uses a song's properties to recommend music with similar properties. Here, we look at content instead of who else is listening to music.

### how can you select k for k-means ? 
---
We use the elbow method to select k for k-means clustering . the idea of the elbow method is to run k-means clustering on the dataset where k is the number of clusters.<br>
within the sum of squares wss it is defined as the sum of the squared distance between each member of the cluster and its centroid. 

### what is the significance of p-value : 
---
p-values typiccally <= 0.05 : this indicates strong evidence against the null hypothesis ; so you reject the null hypothesis. <br>
p-value typically > 0.05 : this indicates weak evidence against the null hypothesis ; so you accept the null hypothesis 

### how can outlier values be treated : 
--- 
- You can drop outliers only if it is a garbage value. example height = ABC 
- If the outliers have extreme values, they can be removed. For example, if all the data points are clustered between zero to 10, but one point lies at 100, then we can remove this point.
- If you cannot drop outliers, you can try the following
	- Try a different model. Data detected as outliers by linear models can be fit by nonlinear models. Therefore, be sure you are choosing the correct model.
	- Try normalizing the data. This way, the extreme data points are pulled to a similar range.
	- You can use algorithms that are less affected by outliers; an example would be [random forests](https://www.simplilearn.com/tutorials/data-science-tutorial/random-forest-in-r "random forests").

### how can time series data be declared as stationery ? 
---
It is stationary when the variance and mean of the series are constant with time.