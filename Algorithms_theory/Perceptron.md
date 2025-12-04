The perceptron is type of neural network that performs binary classification that maps input features to an output decision , usually classifying data into one of two categories such as 0 or 1 . 
Perceptron consists of a single layer of input nodes that are fully connected to a layer of output nodes. It is particularly good at learning linearly separable patterns.  It utilizes a variation of artificial neurons called **Threshold Logic Units (TLU)**, which were first introduced by McCulloch and Walter Pitts in the 1940s. This foundational model has played a crucial role in the development of more advanced neural networks and machine learning algorithms.

## type of perceptron : 
---
1. [**Single-Layer Perceptron**](https://www.geeksforgeeks.org/python/single-layer-perceptron-in-tensorflow/) is a type of perceptron is limited to learning linearly separable patterns. It is effective for tasks where the data can be divided into distinct categories through a straight line. While powerful in its simplicity, it struggles with more complex problems where the relationship between inputs and outputs is non-linear.
2. [**Multi-Layer Perceptron**](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/) possess enhanced processing capabilities as they consist of two or more layers, adept at handling more complex patterns and relationships within the data.
## basic components of perceptron : 
---
A perceptron is composed of key components that work together to process information and make predictions. 
- Input features : the perceptron takes multiple input features each representing a characteristic of the input data 
- Weights : each input feature is assigned a weight that determines its influence on the output. Thes weights are adjusted during training to find the optimal values. 
- Summation Function : the Perceptron calculates the weighted sum of its inputs combining them with their respective weights.
- Activation function : the weighted sum is passed through the Heaviside step function , comparing it to a threshod to produce a binary output (0 or 1)
- ***Output:** The final output is determined by the activation function, often used for **binary classification*** tasks.
- Bias :The bias term helps the perceptron make adjustments independent of the input, improving its flexibility in learning.
- Learning algorithm : the perceptron adjusts its weights and bias using a learning algorithm , such as the Perceptron Learning Rule to minimize prediction errors.

These components enable the perceptron to learn from data and make predictions. While a single perceptron can handle simple binary classification , complex tasks require multiple perceptrons organized into layers , forming a neural network.

## How does Perceptron work ? 
---
A weight is assigned to each input node of a perceptron indicating the importance of that input in determining the output. The Perceptron's output is calculated as a weighted sum of the inputs which is then passed through an activation function to decide whether the perceptron will fire. 
<br>
the weighted sum is computed as : z = w1x1 + ... + wnxn = Xt W
<br>
The step function compares this weighted sum to a threshold. If the input is larger than the threshold value, the output is 1; otherwise, it's 0. This is the most common activation function used in Perceptrons are represented by the Heaviside step function. <br>
A perceptron consists of a single layer of Threshold logic Units (TLU) , with each TLU fully connected to all input nodes. <br>
In a fully conencted layer , also known as a dense layer , all neurons in one layer are connceted to every neuron in the previous layer. 
