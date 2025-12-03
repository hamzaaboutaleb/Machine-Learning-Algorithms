### 1. definition : 
---
is a machine learning classification algorithm that predicts the category of a data point using probability. <br>
It assumes that all features are independent of each other. Naive bayes performs well in many real world applications such as spam filtering , document categorization and sentiment analysis.

### 2. Key features of Naive bayes classifiers : 
---
the main idea behind the algorithm is to use Bayes' Theorem to classify data based on the probabilities of different classes given the features of the data. It is used mostly in high-dimensional text classification.
- The Naive Bayes Classifier is a simple probabilistic classifier and it has very few number of parameters which are used to build the ML models that can predict at a faster speed than other classification algorithms.
- It is a probabilistic classifier because it assumes that one feature in the model is independent of existence of another feature. In other words, each feature contributes to the predictions with no relation between each other.
- Naive Bayes Algorithm is used in spam filtration, Sentimental analysis, classifying articles and many more.

### 3. why it is callsed naive bayes ? 
---
It is named as "Naive" because it assumes the presence of one feature does not affect other features. The Bayes part of the name referes to its basis in Bayes Theorem.

### 4. Assumption of Naive Bayes : 
---
The fundmental Naive Bayes assumption is that each feature makes an : 
- Feature independence : This means that when we are trying to classify something, we assume that each feature (or piece of information) in the data does not affect any other feature.
- Continuous features are normally distributed : if a feature is continuous then it is assumed to be normally distributed within each class 
- Discrete features have multinomial distributions : If a feature is discrete , then it is assumed to have a multinomial distribution within each class
- Features are equally important : all features are assumed to contribute equally to the prediction of the class label 
- No missing data : the data should not contain any missing values

### 5. Introduction to Bayes Theorem : 
---
 Bayes theorem provides a principled way to reverse conditional probabilities. It is defined as : 
 P(y∣X)=P(X∣y)⋅P(y)​ / P(X) <br>Where:

- P(y∣X): Posterior probability, probability of class yy given features XX
- P(X∣y): Likelihood, probability of features XX given class yy
- P(y): Prior probability of class yy
- P(X): Marginal likelihood or evidence

### 6 . Naive Bayes working: 
---
### 1. Terminology : 
---
consider a classification problem (like predicting if someone plays golf based on weather) . then : 
- y is the class label : Yes or No 
- X = (x1...xn) is the feature vector (outlook , temperature , humidity , wind)

A sample row from the dataset:

	 X=(Rainy, Hot, High, False),y=No
this represents : what is the probability that someone will not play the golf given that the weather is rainy hot , high humidity and no wind ? 

### 2. The Naive assumption : 
---
the "naive" comes from the assumption that all features are independent given the class. That is : <br>
$$
P(y \mid x_1, \ldots, x_n)
= \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}{P(x_1, \ldots, x_n)}
$$
Since the denominator is constant for a given input, we can write:
$$
P(y \mid x_1, \ldots, x_n) \propto P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y)
$$
### 3. constructing the naive bayes classifirer :
---
 We compute the posterior for each class y and choose the class with the highest probability:
 $$
\hat{y} = \arg\max_{y} \; P(y) \cdot \prod_{i=1}^{n} P(x_i \mid y)
$$
This becomes our Naive Bayes classifier.


## 7 . Types of Naive Bayes Model : 
---
There are three types of Naive Bayes Model :
### 1 - Gaussian Naive BayesIn
[****Gaussian Naive Bayes****](https://www.geeksforgeeks.org/machine-learning/gaussian-naive-bayes/), continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution. A Gaussian distribution is also called [Normal distribution](https://www.geeksforgeeks.org/maths/normal-distribution/) When plotted, it gives a bell shaped curve which is symmetric about the mean of the feature values as shown below:

### ****2. Multinomial Naive Bayes****

[****Multinomial Naive Bayes****](https://www.geeksforgeeks.org/machine-learning/multinomial-naive-bayes/)is used when features represent the frequency of terms (such as word counts) in a document. It is commonly applied in text classification, where term frequencies are important.

### ****3. Bernoulli Naive Bayes****

[****Bernoulli Naive Bayes****](https://www.geeksforgeeks.org/machine-learning/bernoulli-naive-bayes/) deals with binary features, where each feature indicates whether a word appears or not in a document. It is suited for scenarios where the presence or absence of terms is more relevant than their frequency. Both models are widely used in document classification tasks


## Advantages : 
---
- Easy to implement and computationally efficient.
- Effective in cases with a large number of features.
- Performs well even with limited training data.
- It performs well in the presence of categorical features.
- For numerical features data is assumed to come from normal distributions

## Disadvantages : 
---
- Assumes that features are independent, which may not always hold in real-world data.
- Can be influenced by irrelevant attributes.
- May assign zero probability to unseen events, leading to poor generalization.

### Real world applications : 
---
- ***Spam Email Filtering**: Classifies emails as spam or non-spam based on features.
- ***Text Classification**: Used in sentiment analysis, document categorization and topic classification.
- **Medical Diagnosis:**** Helps in predicting the likelihood of a disease based on symptoms.
- ***Credit Scoring:*** Evaluates creditworthiness of individuals for loan approval.
- ***Weather Prediction**: Classifies weather conditions based on various factors.