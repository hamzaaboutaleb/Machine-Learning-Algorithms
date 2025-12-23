Support Vector Machine (SVM) is a supervised machine learning algorithm used for classification and regression tasks. It tries to find the best boundary known as hyperplane that separates different classes in the data. It is useful when you want to do binary classification like spam vs. not spam or cat vs. dog.<br>
The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data.

## Key concepts of SVM : 
---
- Hyperplane : a decision boundary separating different classes in feature space and is represented by the equation wx + b = 0 in linear classification.
- Support vectors : the closest data points to the hyperplace , crucial for determining the hyperplane and margin in SVM 
- Margin : the distance between the hyperplane and the support vectors. SVM aims to maximize this margin for better classification perfermance. 
- kernel : A function that maps data to a higher-dimensional space enabling SVM to handle non-linearly separable data.
- -**Hard Margin**: A maximum-margin hyperplane that perfectly separates the data without misclassifications.
- ***Soft Margin**: Allows some misclassifications by introducing slack variables, balancing margin maximization and misclassification penalties when data is not perfectly separable.
- **C**: A regularization term balancing margin maximization and misclassification penalties. A higher C value forces stricter penalty for misclassifications.
- ***Hinge Loss**: A loss function penalizing misclassified points or margin violations and is combined with regularization in SVM.
- **Dual Problem**: Involves solving for Lagrange multipliers associated with support vectors, facilitating the kernel trick and efficient computation.
### how does support vector machine algorithm work ? 
---
The key idea behind the SVM algorithm is to find the hyperplane that best separates two classes by maximizing the margin between them. This margin is the distance from the hyperplane to the nearest data points (support vectors) on each side.<br>
The best hyperplane also known as the ****"hard margin"**** is the one that maximizes the distance between the hyperplane and the nearest data points from both classes. This ensures a clear separation between the classes

