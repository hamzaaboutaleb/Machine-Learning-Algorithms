
## 1 - what is the difference between descriptive and inferential statistics ? 
---
- Descriptive statistics aims to summarize and present the features of a giving dataset while inferential statistics leverages sample data to make estimates or test hypotheses on larger population 
### Descriptive statistics : 
---
Descriptive statistics describe the key aspects or characteristics of a dataset:
- Measures of central tendency : Identify central of typical values in the dataset typically using the mean , median or the mode.
- Measures of spread or dispersion : Indicate the variability or spread around the central value, often quantified by the range, standard deviation, or variance.
- Data Distribution : Categorizes the data distribution as normal , skewed or otherwise and assists in virual representation 
- Shape of Data : Describes whether the data is symmetrical or skewed and the extent of that skewness 
- Correlation : measures the relationship or lack thereof between two variables 
- Text statistics : summarizes verbal or written data using word frequencies , readabilities etc ..
### Inferential statistics : 
---
Inferential extends finding from a subset of data to make inferences about an entire population. 
- Hypothesis testing : allows researchers to compare data to an assumed or expected distribution , indicating whether a finding is likely due to chance or not .
- Confidence intervals : Provides a range within which the true population value is likely to fall 
- Regression analysis : predicts the values of dependent variables using one or more independent variables 
- Probability : helps measure uncertainty and likelihood , formint the basis for many inferential statistical tools 
- Sampling techniques : guides researchers in selecting appropriate samples to generalize findings to a wider population 

## 2 . Define and distinguish between population and sample in statistics 
---
the population is the set that contains all the individuals or items that are of interest to a researcher , by constrat a sample is a subset of the population that is selected for analysis.
### notable sampling techniques : 
---
- **Simple Random Sampling**: All individuals in the population have an equal chance of being selected.
    
- **Stratified Sampling**: The population is divided into distinct subgroups, or strata, and individuals are randomly sampled from each stratum.
    
- **Cluster Sampling**: The population is divided into clusters, and then entire clusters are randomly selected.
    
- **Convenience Sampling**: Individuals are chosen based on their ease of selection. This method is usually less rigorous and can introduce sampling bias.
    
- **Machine Learning Connection**: Datasets used for training and testing ML models often represent samples from a larger population. The model's goal is to make predictions about the overall population based on the patterns it identifies in the training sample.

## 3.  Explain what a "_distribution_" is in statistics, and give examples of common distributions.
---
In statistics, a **distribution** describes **how the values of a random variable are spread out** — that is, it tells us **which values are more likely and which are less likely**.

## what are standard deviation and variance ?
---
Variance and standard deviation both measure the dispersion of spread of a dataset. Variance is the average of the squared differences from the mean. It gives a sense of how much the values in a dataset differ from the mean. 
However, because it uses squared difference , the units are squared as well , which can less intuitive than the standard deviation. _Standard deviation is the square root of the variance, bringing the units back to the same as the original data. It provides a more interpretable measure of spread. For example, if the variance of a dataset is 25, the standard deviation is √25 = 5._ 
### what is skewness ? 
---
skewness measures the asymmetry of a dataset around its mean , which can be positive, negative, or zero. Data with positive skewness , or right skewed data , has a longer right tail , meaning  the mean is greater than the median. Data with negative skewness or left skewed data has a longer left tail , meaning the mean is less than the median .Zero skewnedd indicates a symmetric distribution , like a normal distribution where the mean the median and the mode are equals.

## What is a histogram : 
---
A histogram is a graphical representation of the distribution of a dataset, it divides the data into bins