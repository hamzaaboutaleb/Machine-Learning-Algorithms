# Machine learning pipeline 
---
# 1 . data preprocessing 
---
## 1.1 ML Workflow 
---
ML lifecycle is a structered process that defines how machine learning models are developed , deployed and maintained. It consists of a series of steps that ensure the model is accurate , reliable and scalable.<br>
It includes defining the problem , collecting and preparing the data , exploring patterns, engineering features , training and evaluating models, deploying them into production and continuously monitoring performance to handle issues like data drift and retraining needs. Below are the key steps of the ML lifecycle :
### step 1 : Problem definition : 
---
The first step is identifying and clearly defining the business problem, A well framed problem provides the foundation for the entire lifecycle. Important things like project objectives , desired outcomes and the scope of the tast are carefully designed during this stage.
- collaborate with stakeholders to understand business goals . 
- define project objectives, scope and success criteria
- ensure clarity in desired outcomes
### step 2 : Data collection 
---
Data collection phase involves systematic collection of datasets that can be used as raw data to train model. the quality and variety of data directly affect the models performance 

<br>
Here are some basic features of Data collection : 
1. Relevance : collect the data should be relevant to the defined problem and include necessary features 
2. quality : ensure data quality by considering factors like accuracy and ethical use
3. quantity : gather sufficient data volume to train a robust model 
4. diversity : include diverse datasets to capture a broad range of scenarios and patterns

### step 3 : data cleaning and preprocessing 
---
Raw data is often messy and unstructured and if we use this data directly to train then it can lead to poor accuracy.<br>
we need to do __data cleaning and preprocessing__ which often involves : 
- Data cleaning : address issues such as missing values , outliers and inconsistencies in the data.
- Data preprocessing : standardize formats , scale values and encode categorical variables for consistency.
- Data quality : ensure that the data is well organized and prepared for meaningful analysis.

### Step 4 : Exploratory data analysis EDA 
---
to find patterns and characteristics hidden in the data EDA is used to uncover insights and understand the dataset's structure. During EDA patterns, trends and insights are provided which may not be visible by naked eyes. This valuable insight can be used to make informed decision.<br>
Here are the basic features of Exploratory Data Analysis:

- ***Exploration:** Use statistical and visual tools to explore patterns in data.
- ***Patterns and Trends:** Identify underlying patterns, trends and potential challenges within the dataset.
- ***Insights:** Gain valuable insights for informed decisions making in later stages.
- **Decision Making:** Use EDA for feature engineering and model selection.
### Step 5 : feature engineering and Selection 
---
It is a transformative process that involve selecting only relevant features to enhance model efficiency and prediction while reducing complexity.
the basic features of Feature engineering and selection :
 - Feature engineering : create new features or transform existing ones to capture better patterns and relationships 
 - Feature selection : identify subset of features that most significantly impact the models performance
 - Domain expertise : use domain knowledge to engineer features that contribute meaningfully for prediction 
 - optimization : balance set of features for accuracy while minimizing computational complexity 

### step 6 : Model selection 
---
for a good machine learning model , model selection is a very important part as we need to find model that aligns with our defined problem, nature of the data , complexity of problem and the desired outcomes. The basic features of model selection : 
- complexity : consider the complexity of the problem and the nature of the data when choosing a model.
- Decision factors : Evaluate factors like performance, interpretability and scalability when selecting a model.
- Experimentation :Experiment with different models to find the best fit for the problem.

### Step 7 : Model training : 
---
With the selected model the machine learning lifecycle moves to model training process. This process involves exposing model to historical data allowing it to learn patterns , relationships and dependencies within the dataset. <br>
Here are the basic features of model training : 
- Iterative process : Train the ,odel iteratively , adjusting paramaters to minimize errors and enhance accuracy. 
- Optimization : fine tune model to optimize its predictive capabilities 
- Validation : rigorously train model to ensure accuracy to new unseen data.

### step 8 : model evaluation and tuning : 
---
Model evaluation involves rigorous testing against validation or test datasets to test accuracy of the model on new unseen data, It provides insights into models strengths and weaknesses. If the model fails to achieve desired performance levels we may need to tune model again and adjust its hyperparameters to enhance predictive accuracy. The basic features of model evaluation and tuning : 
- Evaluation metrics : use metrics like accuracy , precision recall and F1 score to evaluate model performace. 
- Strenghts and weaknesses : identify the strenghts and weaknesses of the model through rigorous testing.
- Iterative improvement : initiate model tuning to adjust hyperparameters and enhance predictive accuracy
- Model Robustness : iterative tuning to achieve desired levels of model robustness and reliability 

### Step 9 : Model Deployment
---
Now the model is ready to be deployed for real world application . It involves intefrating the predictive model with existing systems allowing business to use this for informed decision making.
The basic features of model deployment : 
- Integrate with existing system 
- Enable decision making using predictions 
- Ensure deployment scalabity and security
- Provide APIs or pipelines for production use

### Step 10 : model monitoring and maintenance : 
---
After deployment models must be monitored to ensure they perform well over time , Regular tracking helps detect data drift , accuracy drops or changing patterns and retraining may be needed to keep the model reliable in real world use. 
basic features : 
- Track model performace over time 
- Detect data drift or concept drift
- Updata and retrain the model when accuracy drops 
- Maintain logs and alerts for real time issues
Each step is essential for building a successful machine learning model that can provide valuable insights and predictions. By following the Machine learning lifecycle organizations we can solve complex problems.


## 1.2 Data cleaning : 
---
Data cleaning is a step in machine learning which involves identifying and removing any missing , duplicate , or irrelevant data .
- Raw data (log file, transactions, audio /video recordings, etc) is often noisy, incomplete and inconsistent which can negatively impact the accuracy of model.
-  The goal of data cleaning is to ensure that the data is accurate, consistent and free of errors.
-  Clean datasets also important in EDA (Exploratory Data Analysis) which enhances the interpretability of data so that the right actions can be taken based on insights.
### how to perform data cleaning :
---
The process begins by identifying issues like missing values , duplicates and outliers. Performing data cleaning involves a systematic process to identify and remove errors in a dataset.The following steps are essential to perform data cleaning:
- Remove unwanted observations : eliminate duplicates,  irrelevant entries or redundant data that add noise. 
- Fix Structural errors : Standardize data formats and variable types for consistency.
- Manage outliers : detelect and handle extreme values that can skew results either by removal or transformation.
- Handle Missing data : address gaps using imputation , deletion or advanced techniques to maintain accuracy and integrity 

### Implementation for data cleaning : 
---
#### step 1 : import librairies and load dataset : 
---
You need to start by importing important librairies and load the dataset (numpy and pandas etc)

#### step 2 : check for duplicate Rows 
---
df.duplicated() : returns a boolean searies indicating duplicate rows

#### Step 3 : identify column data types : 
---
- List comprehension with .dtype attribute to separate categorical and numerical columns. 
- object dtype : Generally used for text or categorical data.

#### step 4 : count unique values in the categorical columns : 
---
***df[numeric_columns].nunique():** Returns count of unique values per column.

#### step 5 : calculate missing values as pourcentage : 
---
- ***df.isnull():** Detects missing values, returning boolean DataFrame.
- Sum missing across columns, normalize by total rows and multiply by 100.

#### step 6 : drop irrelevant or data heavy missing column : 
---
- ***df.drop(columns=[])**: Drops specified columns from the DataFrame.
- ***df.dropna(subset=[])**: Removes rows where specified columns have missing values.
- ***fillna()**: Fills missing values with specified value (e.g., mean).

#### Step 7 : Detect outliers with Box Plot : 
---
- ***matplotlib.pyplot.boxplot():** Displays distribution of data, highlighting median, quartiles and outliers.
- ***plt.show()**: Renders the plot.

#### step 8 : calculate Oultlier Boundaries and remove them 
---
- Calculate mean and standard deviation (std) using df['Age'].mean() and df['Age'].std().
- Define bounds as mean ± 2 * std for outlier detection.
- Filter DataFrame rows within bounds using Boolean indexing.

#### step 9 : input missing data if Any : 
---
fillna() applied again on filtered data to handle any remaining missing values.  (example fill the missing rows of Age column with the mean)

#### step 10 : Recalculate outlier Bounds and remove outliers from the updated data : 
----
- ***mean = df3['Age'].mean()**: Calculates the average (mean) value of the Age column in the DataFrame df3.
- ***std = df3['Age'].std()**: Computes the standard deviation (spread or variability) of the Age column in df3.
- ***lower_bound = mean - 2 * std**: Defines the lower limit for acceptable Age values, set as two standard deviations below the mean.
- ***upper_bound = mean + 2 * std**: Defines the upper limit for acceptable Age values, set as two standard deviations above the mean.
- ***df4 = df3[(df3['Age'] >= lower_bound) & (df3['Age'] <= upper_bound)]**: Creates a new DataFrame df4 by selecting only rows where the Age value falls between the lower and upper bounds, effectively removing outlier ages outside this range.

#### step 11 : data validation and verification : 
---
Data validation and verification involve ensuring that the data is accurate and consistent by comparing it with external sources or expert knowledge. For the machine learning prediction we separate independent and target features. Here we will consider only 'Sex' 'Age' 'SibSp', 'Parch' 'Fare' 'Embarked' only as the independent features and Survived as target variables because PassengerId will not affect the survival rate.

#### step 12 : data formating : 
---
Data formatting involves converting the data into a standard format or structure that can be easily processed by the algorithms or models used for analysis. Here we will discuss commonly used data formatting techniques i.e. Scaling and Normalization.

Scaling involves transforming the values of features to a specific range. It maintains the shape of the original distribution while changing the scale. It is useful when features have different scales and certain algorithms are sensitive to the magnitude of the features.


## 1.3 . Feature Engineering : Scaling , normalizaiton and standardization  
---
Feature engineering is the process of creating , transforming or selecting the most relevant variables from raw data to improve model performance. Effective features help the model capture important patterns and relationships in the data . It directly contributes to model building in the following ways : 
- Well-designed features allow models to learn complex patterns more effectively.
- Reduces noise and irrelevant information , improving prediction accuracy. 
- Helps prevent overfitting by emphasizing meaningful data signals.
- Simplifies model interpretation by creating more informative and understandable inputs.
There are various techniques such as scaling , normalization and standardizaation that can be used for feature engineering.

### 1. absolute maximum scaling : 
---

Absolute Maximum Scaling rescales each feature by dividing all values by the maximum absolute value of that feature. This ensures the feature values fall within the range of -1 to 1. While simple and useful in some contexts, it is highly sensitive to outliers  which can skew the max absolute value and negatively impact scaling quality.

### 2. Min max scaling : 
---
Min-Max Scaling transforms features by subtracting the minimum value and dividing by the difference between the maximum and minimum values. This method maps feature values to a specified range, commonly 0 to 1, preserving the original distribution shape but is still affected by outliers due to reliance on extreme values.
- Scales features to range.
- Sensitive to outliers because min and max can be skewed.

### 3. Normalization (vector normalization)
---
Normalization scales each data sample (row) such that its vector length (Euclidean norm) is 1. This focuses on the direction of data points rather than magnitude making it useful in algorithms where angle or cosine similarity is relevant, such as text classification or clustering.


### 4. Standardization : 
---
Standardization centers features by subtracting the mean and scales them by dividing by the standard deviation, transforming features to have zero mean and unit variance. This assumption of normal distribution often benefits models like linear regression, logistic regression and neural networks by improving convergence speed and stability.

### 5. Robust scaling : 
---
Robust Scaling uses the median and interquartile range (IQR) instead of the mean and standard deviation making the transformation robust to outliers and skewed distributions. It is highly suitable when the dataset contains extreme values or noise.

###  Comparison of Various Feature Scaling Techniques
---

|Type|Method Description|Sensitivity to Outliers|Typical Use Cases|
|---|---|---|---|
|Absolute Maximum Scaling|Divides values by max absolute value in each feature|High|Sparse data, simple scaling|
|Min-Max Scaling (Normalization)|Scales features to by min-max normalization|High|Neural networks, bounded input features|
|Normalization (Vector Norm)|Scales each sample vector to unit length (norm = 1)|Not applicable (per row)|Direction-based similarity, text classification|
|Standardization (Z-Score)|Centers features to mean 0 and scales to unit variance|Moderate|Most ML algorithms, assumes approx. normal data|
|Robust Scaling|Centers on median and scales using IQR|Low|Data with outliers, skewed distributions|
### Advantages : 
---
- Improve model performance : Enhances accuracy and predictive power by presenting features in comparable scales .
- Speeds up convergence : helps gradient based algorithms train faster and more reliably 
- Prevents feature bias : avoid dominance of large scale features.
- Increases numerical stability : reduces risks of overflow underflow in computations . 
- Facilitates algorithm compatibility : makes data suitable for distance and gradient based models like SVM , KNN and neural networks 


##  Features engineering : 
---
### Process involved in FE : 
---
1. Feature creation : Feature creation involves generating new features from domain knowledge or by observing patterns in the data. It can be:
	1. **Domain-specific**: Created based on industry knowledge like business rules.
	2. **Data-driven**: Derived by recognizing patterns in data.
	3. **Synthetic**: Formed by combining existing features.
2. Feature transformation : transformation adjusts feature to improve model learning : 
	 1. **Normalization & Scaling**: Adjust the range of features for consistency.
	2. **Encoding**: Converts categorical data to numerical form i.e one-hot encoding.
	3. **Mathematical transformations**: Like logarithmic transformations for skewed data.
3. Feature Extraction : Extracting meaningful features can reduce dimensionality and improve model accuracy : 
	1. Dimensionality Reduction : techniques like PCA reduce features while preserving important information.
	2. Aggregation & combination : summing or averaging features to simplify the model 
4. Feature Selection : involves chosing a subset of relevant features to use : 
	 - **Filter methods**: Based on statistical measures like correlation.
	- **Wrapper methods**: Select based on model performance.
	- **Embedded methods**: Feature selection integrated within model training.
5. Feature Scaling : scaling ensures that all features contribute equally to the model. 

### Steps in Feature engineering : 
---

1. **Data Cleaning:** Identify and correct errors or inconsistencies in the dataset to ensure data quality and reliability.
2. **Data Transformation:** Transform raw data into a format suitable for modeling including scaling, normalization and encoding.
3. **Feature Extraction:** Create new features by combining or deriving information from existing ones to provide more meaningful input to the model.
4. **Feature Selection:** Choose the most relevant features for the model using techniques like correlation analysis, mutual information and stepwise regression.
5. **Feature Iteration:** Continuously refine features based on model performance by adding, removing or modifying features for improvement.


## Feature Selection : 
---
Feature selection is the process of choosing only the most useful input features for a machine learning model. It helps improve model performance, reduces noise and makes results easier to understand.

- Helps remove irrelevant and redundant features
- Improves accuracy and reduces overfitting
- Speeds up model training
- Makes models simpler and easier to interpret

### Need of Feature Selection : 
---
Feature selection methods are essential in data science and machine learning for several key reasons:
- ****Improved Accuracy****: Models learn better when trained on only important features.
- ****Faster Training****: Fewer features reduce computation time.
- ****Greater Interpretability****: With fewer inputs, understanding model behavior becomes easier.
- ****Avoiding the Curse of Dimensionality****: Reduces complexity when working with high-dimensional data.

### Type of feature Selection methods : 
---
There are various algorithms used for feature selection and are grouped into three main categories and each one has its own strengths and trade-offs depending on the use case.
### 1. Filter Methods
---
it evaluate each feature independently with target variable. Feature with high correlation with target variable are selected as it means this feature has some relation and can help us in making predictions. These methods are used in the preprocessing phase to remove irrelevant or redundant features based on statistical tests (correlation) or other criteria.<br>
__common Filter techniques__ :
- Information Gain : Measures reduction in entropy when a feature is used. 
- Chi-square test : Checks the relationship between categorical features. 
- Fishers score : Ranks features based on class separability.
- Variance threshold : removes features with very low variance 
- Dispersion ration : ration of arithmetic mean to geometric mean , higher values indicate useful features 

__advantages__ : 
- fast and efficient : filter methods are computationally inexpensive, making them ideal for large datasets.
- Easy to implement :These methods are often built-in to popular machine learning libraries, requiring minimal coding effort.
- Model independence : Filter methods can be used with any type of machine learning model, making them versatile tools.

__Limitations__ : 
- Limited interaction with the model : since they operate independetly , filter methods might miss data interactions that could be important for prediction 
- Choosing the right metric  : Selecting the appropriate metric for our data and task is important for optimal performance.

### 2. Wrapper methods : 
---


Wrapper methods are also referred as greedy algorithms that train algorithm. They use different combination of features and compute relation between these subset features and target variable and based on conclusion addition and removal of features are done. Stopping criteria for selecting the best subset are usually pre-defined by the person training the model such as when the performance of the model decreases or a specific number of features are achieved.

**Common Wrapper Techniques**

- forward selection : start with no features and add one at a time based on improvement 
- Backward elimination : start with all features and remove the least useful ones 
- Recursive feature eliminatin RFE : removes the least important features step by step 

__Advantages__ : 
- Model specific optimization : wrapper methods directly consider how features influence the model , potentially leading to better performance compared to filter methods 
- Flexible these methods can be adapted to various model types and evaluation metrics 

__Limitations__ : 
- Computationally expensive :Evaluating different feature combinations can be time-consuming, especially for large datasets.
- Risk of overfitting : fine tuning features to a specific model can lead to an overfitted model that performs poorly on unseen data 


###  3. Embedded methods
---
it perform feature selection during the model training process. They combine the benefits of both filter and wrapper methods. Feature selection is integrated into the model training allowing the model to select the most relevant features based on the training process dynamically.
**Common Embedded Techniques** : 
- L1 regularization (Lasso) : keeps only the features with non zero coefficients 
- Decision trees and Random forests : select features based on impurity reduction
- Gradient boositng :pick features that reduce prediction error the most 

__advantages :__
- efficient and effective : embedded methods can achieve good results without the computational burden of some wrapper methods 
- Model specific learning : similar to wrapper methods these techniques uses the learning process to identify relevant features 

__Limitations__
- Limited interpretability : Embedded methods can be more challenging to interpret compared to filter methods making it harder to understand why specific features were chosen 
- Not universally applicable : not all machine learning algorithms support embedded feature selection techniqeus 


### choosing the right FS method : 
---

Choice of feature selection method depends on several factors:

- **Dataset size**: Filter methods are generally faster for large datasets while wrapper methods might be suitable for smaller datasets.
- **Model type**: Some models like tree-based models, have built-in feature selection capabilities.
- **Interpretability**: If understanding the rationale behind feature selection is crucial, filter methods might be a better choice.
- **Computational resources:** Wrapper methods can be time-consuming, so consider our available computing power.

With these feature selection methods we can easily improve performance of our model and reduce its computational cost.


## Exploratory Data Analysis : 
---
EDA is an important step in data science and data analystics as it visualizes data to understand its main features , find patterns and discover how different parts of the data are connected. 

### Type of EDA: 
---
### 1.1 . Univariate analysis : 
---
It focuses on studying one variable to understand its characteristics. It helps to describe data and find patterns within a single feature. Various common methods like histograms are used to show data distribution , box plots to detect outliers and understand data spread and bar charts for categorical data. summary statistics like mean , median , mode , variance and std helps in describing the central tendency and spread of the data
### 1.2 Bivariate analysis : 
---
it docuses on identifying relationship between two variables to find connections correlations and dependencies. It helps to understand how two variables interact with each other. Some key techniques includes : 
- Scatter plots which visualize the relationship between two continuous variables. 
- Correlation coefficient measutes how strongly two variables are related which commonly use Pearson's correlation for linear relationships. 
- Cross Tabulations or contingency tables shows the frequency distribution of two categorical variables and help to understand their relationship.
- Line  graphs are useful for comparing two variables over time in time series data to identify trends or patterns
- Covariance measures how two variables change together but it is paired with the correlation coefficient for a clearer and more standardized understanding of the relationship

### 1.3 Multivariate Analysis : 
---
it identify relationships between two or more variables in the dataset and aims to understand how variables interact with one another which is important for statistical modeling techniques . it include techniques like : 
- Pair plots which shows the relationships between multiple variables at once and helps in understanding how they interact
- PCA : reduces the complexity of large datasets by simplifying them while keeping the most important information 
- Spatial Analysis : used for geographical data by using maps and spatial plotting to understand the geographical distribution of variables.
- Time series analysis : used for datasets that involve time based data and it involves understanding and modeling patterns and trends over time. Common techniques include line plots , auto correlation analystis , moving averages and ARIMA models

### steps to perform EDA : 
---
It involves a series of steps to help us understand the data, uncover patterns, identify anomalies, test hypotheses and ensure the data is clean and ready for further analysis

### step 1 : understand the problem and the data : 
---
the first step of any data analysis problem starts by understanding well the problem we re trying to solve and the data we have , and that include asking the following questions : 
- what is the business goal or research question ? 
- what are the variables in the data and what do they represent ? 
- what types of data(numerical , categorical , text , etc. ) do you have ? 
- are there any known data quality issues or limitations ? 
- are there any domain specific concerns or restrictions ? 
by understanding the problem and the data we can plan our analysis more effectively , avoid incorrect assumptions and ensure accurate conclusions. 

### step 2 : importing and inspecting the data
---
After understanding the problem and the data, next step is to import the data into our analysis environment such as Python, R or a spreadsheet tool. It’s important to find data to gain an basic understanding of its structure, variable types and any potential issues. Here’s what we can do:
1. Load the data into our environment carefully to avoid errors or truncations.
2. Check the size of the data like number of rows and columns to understand its complexity.
3. Check for missing values and see how they are distributed across variables since missing data can impact the quality of your analysis.
4. Identify data types for each variable like numerical, categorical, etc which will help in the next steps of data manipulation and analysis.
5. Look for errors or inconsistencies such as invalid values, mismatched units or outliers which could show major issues with the data.
by completing the tasks above you ll be ready to do the data cleaning 

### step 3 : handling missing data : 
---
Missing data is common in many datasers and can affect the quality of our analysis. During EDA its important to identify and handle missing data properly to avoid biased or misleading results. Heres how to handle it : 
1. Understand the patterns and possible causes of missing data. Is it missing completely at random (MCAR) , missing at random (MAR) or not missing at random (MNAR) , identifying this helps us to fund the best way to handle the missing data.
2. Decide whether to remove missing data or impute the missing values ,Removinf data can lead to biased outcomes if the missing data isnt MCAR. Filling values helps to preserve data but should be done carefully 
3. Use appropriate imputation methods like mean or median imputation , regression imputation or machine learning techniques like KNN , decision trees based on the data s characteristics
4. Consider the impact of missing data. even after imputting missing data can cause uncertainty and bias so understands the result with caution. 
Properly handling of missing data improves the accuracy of our analysis and prevents misleading conclusions.

### step 4 : exploring data characteristics : 
---
After addressing missing data we find the characteristics of our data by checking the distribution, central tendency and variability of our variables and identifying outliers or anomalies. This helps in selecting appropriate analysis methods and finding major data issues. We should calculate summary statistics like mean, median, mode, standard deviation, skewness and kurtosis for numerical variables. These provide an overview of the data’s distribution and helps us to identify any irregular patterns or issues.

### step 5 : perform data transformation 
---
Data transformation is an important step in EDA as it prepares our data for accurate analysis and modeling.Depending on our data's characteristics and analysis needs, we may need to transform it to ensure it's in the right format. commmon transformation techniques include : 
1. Scaling or normalization 
2. encoding categorical variables 
3. Applying mathematical transformation like Log square root 
4. creating new variables from existing ones like calculating rations or combining columns 
5. aggrefating or grouping data based on one specific variables or conditions 

### Step 6 : Visualizing relationship of data
---
Visualization helps to find relationshops between variables and identify patterns or trends that may not be seen from the summary alone. 
1. For categorical data : create frequency tables , bar plots , pie charts to understand the distribution of categories and identify imblanaces on unusual patterns .
2. For numerical variabels generate histograms, box plots , violing plots and density plots to visualize distribution , shape , spread and porential outliers. 
3. To find relationships between variables use scatter plots , correlation matrices or statistical tests like Pearson's correlation coefficient or Spearmans's rank correlation 

### Step 7: handling outliers: 
---
Outliers are data points that differs from the rest of the data may caused by errors in measurement or data entry.Detecting and handling outliers is important because they can skew our analysis and affect model performance.We can identify outliers using methods like IQR , Z-scores or domain-specific rules. Once identified it can be removed or adjusted depending on the context. Properly managing outliers shows our analysis is accurate and reliable.

### step 8 : communicate findings and Insights : 
---
The final step in EDA is to communicate our findings clearly. This involves summarizing the analysis, pointing out key discoveries and presenting our results in a clear way.
1. Clearly state the goals and scope of your analysis.
2. Provide context and background to help others understand your approach.
3. Use visualizations to support our findings and make them easier to understand.
4. Highlight key insights, patterns or anomalies discovered.
5. Mention any limitations or challenges faced during the analysis.
6. Suggest next steps or areas that need further investigation.
Effective communication is important to ensure that our EDA efforts make an impact and that stakeholders understand and act on our insights. By following these steps and using the right tools, EDA helps in increasing the quality of our data, leading to more informed decisions and successful outcomes in any data-driven project.


## Advanced EDA : 
---
### 1. Understanding the basics of Descriptive statistics : 
---
Descriptive statistics give us a clear picture of the distribution, spread and central tendency of the data.These measures allow us to summarize the data in ways that make it easier to analyze and interpret
### 1.1. Mean : 
---
The mean is the average of the data points, calculated by summing all values and dividing by the total number of observations.
- Best used : The mean is particularly useful when comparing different sets of data that are similar in distribution and dont have extreme values. For instance , comparing the average invome levels across different regions or departments in a company.
- Not suitable : the mean can be heavly influenced by outliers or skewed data. If the dataset contain unusually high values (like a few people earning extremely high incomes) it may distort the results. The mean would no long represent the typical value in this case 
- Example: If we want to understand the average monthly sales of a store over the course of a year, we would calculate the mean sales to see the typical revenue generated each month.

### 1.2 Median : 
---
The median is the middle value of the dataset when arranged in ascending order. It is robust to outliers, meaning that extreme values do not significantly affect the median.
- Best used : The median is ideal for datasets that are skewed or have outliers. It gives a better sense of the "typical" value in cases where the mean may be misleading. For example, when calculating household income in a region where a few individuals earn significantly more than the rest.
- Not suitable : If we're interested in understanding the exact average value, especially when the data distribution is relatively symmetrical, the median may not be ideal. It won’t account for the size of the values, just the middle value.
- example :In a dataset of household incomes, where a few individuals have very high incomes, the median provides a better representation of the typical household income than the mean would.

### 1.3 Mode : 
---
The mode is the most frequent value or category in the dataset.
- Best used : The mode is useful for categorical or discrete data where we want to identify the most common value. For instance, if we want to know the most popular product sold in a store, the mode will give us the product that sold the most units.
- Not suitable : When the data is continuous or doesn’t have a clear frequency, the mode may not provide meaningful insights. For example, continuous data like height or weight typically won’t have a mode.
- Example : A company might want to know which product was sold the most during a promotional campaign. By calculating the mode, they can easily identify the most frequent product sold.

### 1.4 Standard deviation : 
---
Standard deviation measures the amount of variation or dispersion from the mean.A low standard deviation means the data points are close to the mean, while a high standard deviation indicates a greater spread of data points.
- Best used : Standard deviation is useful when we want to understand how spread out the data is. For example, if we're analyzing the daily website traffic for an e-commerce site, a high standard deviation would indicate that traffic varies significantly day-to-day.
- Not suitable : Std can be misleading if the data is heavily skewed or has outliers. In these cases , the standard deviation might not be accurately reflect the true spread of the majority of the data. 
- Example : If an e-commerce website experiences major traffic spikes on certain days, the standard deviation will indicate how much the daily traffic varies from the average, helping to identify whether the site’s traffic is consistent or highly variable.

### 1.5. Interquartile Range (IQR): 
---
The IQR is the difference between the 75th percentile (Q3) and the 25th percentile (Q1) of the data. It represents the spread of the middle 50% of the data and is helpful for identifying outliers.
- Best used : IQR is particularly effective for detecting outliers and understanding the spread of the middle 50% of the data. For instance, when analyzing exam scores in a class, the IQR can help identify students who performed significantly better or worse than most of the class.
- Not suitable :The IQR may not be helpful when the data is already normally distributed, or when there are no outliers in the dataset. In such cases, simpler measures like the mean or standard deviation might be more appropriate.
- Example : In a class of students, if we want to focus on the range of scores that represent the middle 50% of students and exclude extreme values (such as a few students who scored abnormally high or low), we would use the IQR.

### 1.6 Skewness : 
---
Skewness measures the asymmetry of the data distribution. It indicates whether the data leans toward the right (positive skew) or left (negative skew). In simple terms, it tells us whether the data is more on one side than the other.
- best used : When determining if the data needs transformation (such as using a log transform to normalize skewed data). If the data has a significant skew (positive or negative), we might need to apply a transformation to make it more suitable for machine learning algorithms that assume normality (e.g., linear regression).
- not suitable : For symmetric data. If the data is already normally distributed, calculating skewness isn't necessary, as it will be close to zero, offering little additional information.
- example scenario : A retail analyst might use skewness to analyze monthly sales data for a product. If the data is skewed (e.g., higher sales during holiday periods), the analyst may decide to use a log transformation to stabilize variance before applying machine learning models.

### 1.7 Kurtosis : 
---
Kurtosis measures the “tailedness” of the distribution or how extreme outliers are. It tells us whether the data has heavy tails (high kurtosis) or light tails (low kurtosis) compared to a normal distribution. High kurtosis indicates that the data has more extreme outliers than a normal distribution, while low kurtosis suggests fewer extreme values.
- Best used : For identifying datasets with more outliers than expected. High kurtosis might signal that we need to pay attention to outliers, or that the data might be prone to extreme values that could affect the performance of certain models.
- not suitable : For normal data, where the tails are not of particular interest. If a dataset is already fairly well-behaved with a near-normal distribution, kurtosis might not provide additional value.
- example : A risk manager analyzing daily stock returns might calculate kurtosis to identify potential for extreme loss days. If the kurtosis is high, the manager might use techniques to account for those outliers, such as robust statistics or adjusting risk models to reflect the volatility.

### 2. Visualizing distributions : 
---
Visualization is a critical step in EDA, as it helps to identify patterns, trends and anomalies in the data. Selecting the right type of visualization is crucial to gaining meaningful insights.
### 2.1. Bar Plot
---
A bar plot displays the frequency or proportion of categories in categorical data, helping to compare the size of different categories.
![](images/Pasted%20image%2020251223123423.png)