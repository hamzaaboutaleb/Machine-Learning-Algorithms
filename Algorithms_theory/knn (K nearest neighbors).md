### introduction to KNN : 

KNN : is an algorithm of supervised machine learning that been used for prediction or classification problems (is used also in missing values imputation) . 
the algorithm is based on the assumption that similar objects are tend to be found near each others . 
*Example* : You are looking for some crime book in a library lets say "in cold blood" , you probably will look close to other criminal books and not in the cooking area. 

#### what is a classifier ? 

is a type of machine learning models , that take an input and give a category or a label . [example : getting an image and give a cat or a dog]
The goal  of the classifier is to *assigns new , unseen examples to one of the several known class based on what it learned from the labeled data training !*
the classifier understand , learn the patterns between the features and labels and predict the gender of new data points . 


#### how KNN learn ! 

KNN works like the **“memory-based”** learner of machine learning.  
Instead of explicitly learning patterns or building a model, it **remembers all the training data**.
When new data is introduced, the algorithm:

1. **Calculates the distance** between the new data point and all existing points in the training set (using distance metrics like *Euclidean* or *Manhattan*).  
2. **Finds the K nearest neighbors** — the closest data points to the new one.  
3. **Makes a prediction** based on those neighbors:  
   - For **classification**, it takes a **majority vote** of the neighbors’ labels.  
   - For **regression**, it computes the **average** of the neighbors’ values

#### how to chose the right distance metric : 

before lets understand each distance metric : 

1. Euclidean distance : 

   Euclidean distance is defined as the straight-line distance between two points in a plane or space. You can think of it like the shortest path you would walk if you were to go directly from one point to another.
   ![[Screenshot 2025-11-12 205857.png]]

2. Manhatten Distance : 

   This is the total distance you would travel if you could only move along horizontal and vertical lines like a grid or city streets. It’s also called "taxicab distance" because a taxi can only drive along the grid-like streets of a city.
   ![[Screenshot 2025-11-12 210053.png]]
3. Minkowski Distance : 
   Minkowski distance is like a family of distances, which includes both Euclidean and Manhattan distances as special cases.
![[Screenshot 2025-11-12 210103.png]]

From the formula above, when p=2, it becomes the same as the Euclidean distance formula and when p=1, it turns into the Manhattan distance formula. Minkowski distance is essentially a flexible formula that can represent either Euclidean or Manhattan distance depending on the value of p.

A note : all the metrics we mentioned need a scaled data. 
![[Screenshot 2025-11-12 210453.png]]


#### How to choose the value of k for KNN Algorithm?

