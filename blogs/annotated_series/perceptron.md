# <center>The Annotated Neural Network</center>

## <center>[The Perceptron](https://www.ling.upenn.edu/courses/cogs501/Rosenblatt1958.pdf)</center>

#### <center>*Veer Pareek*</center>

The Perceptron is the simplest form of a Neural Network. This post presents an annotated version of the architecture in a line-by-line implementation in python.

### Background

The perceptron, also referred to as the McCulloch-Pitts neuron, is a machine learning algorithm for [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). Specifically, this algorithm works for [binary classification](https://en.wikipedia.org/wiki/Binary_classification). It is also a type of [linear classifier](https://en.wikipedia.org/wiki/Linear_classifier). 

The Perceptron was originally invented in 1943 by Warren McCulloch and Walter Pitts. The first hardware implementation was built in 1957 by Frank Rosenblatt. The Mark I Perceptron had 3 layers:

- An array of 400 photocells arranged in a 20x20 grid, named S-units
- A hidden layer of 512 perceptrons, named A-units
- An output layer of 8 perceptrons, named R-units

Each S-unit can connect up to 40 A-units, stochastically, to eliminate any initial bias. For more details on the hardware implementation, refer to the original paper above.

### Model Architecture
The modern Perceptron consists of the following components:

**Input Layer**

- Consists of *n* input nodes where each node represents a feature of the input data
- A bias input, which always has a value of 1

**Weights**

- Each input has an associated weight
- Weights are adjustable and learned during training

**Single Neuron**

- One output neuron that computes the weighted sum of inputs
- Applies an activation function to weighted sum

**Activation Function**

- A threshold function

**Output**

- A single scalar value, either 0 or 1

You may notice that this modern perceptron differs from Rosenblatt's original implementation. Specifically the lack of A-units. This is because it was proven that the A-units did not provide significant advantages for the types of problems perceptrons are meant to solve. Removing the A-units reduced computational complexity with sacrificing the perceptron's fundamental capabilities. Though, the concept of A-units (intermediate layers) is key for Multi-Layer Perceptrons and other Neural Network variants. 

### The Annotated Implementation
This is an implementation written from scratch. While a single-layer perceptron is rarely used in practice, in general, it is better to use libraries such as PyTorch or TensorFlow for deep learning as they are highly optimized.

##### Dependencies

Being an implementation from scratch, numpy is the only dependency. NumPy is a library which adds support for arrays and matrices with functions to operate on them.
```python
import numpy as np
```

##### Initialization
The `Perceptron` class is initialized with four parameters:

- `num_features` : the number of inputs into the network
- `learning_rate` : scalar value that controls the magnitude of weight updates during training
- `num_epochs` : the number of iterations in trainng
- `weights` : numerical values that deteremine the importance and influence of each input feature, these are adjusted during training

```python
class Perceptron:
	def __init__(self, num_features, learning_rate, num_epochs)
		self.num_features = num_features
		self.learning_rate = learning_rate
		self.num_epochs = num_epochs
		self.weights = np.random.randn(num_features + 1)
```

##### Activation Function
The activation function in a single layer perceptron is the threshold function. Since it is meant for binary classification, it checks whether `x` is above a certain threshold and returns 1 if so, and 0 if not. It is mathematically represented as:

<div style="font-family: 'Times New Roman', Times, serif; font-size: 18px;">
  <p>
    f(x) = 
    {
      <span style="display: inline-block; vertical-align: middle;">
        <span style="display: block; border-bottom: 1px solid black; padding-bottom: 2px;">1 &nbsp; if x ≥ 0</span>
        <span style="display: block; padding-top: 2px;">0 &nbsp; if x < 0</span>
      </span>
    }
  </p>
</div>

This function is implemented in the Perceptron class as:
```python
def activate(self, x):
	return 1 if x >= 0 else 0
```

##### Prediction
The prediction function takes in one parameter
- `inputs` : the input data for prediction

The following line does a few things:
```python
summation = np.dot(np.insert(inputs, 0, 1), self.weights)
```

- `np.insert(inputs, 0, 1` adds a 1 at the beginning of the input array which represents the bias term
- `self.weights` are the current weights of the perceptron, including the bias
- `np.dot` calculates the dot product between the modified inputs and the weights

Finally, the summation is passed through the threshold activation function to return 1 or 0:
```python
return self.activate(summation)
```

The full predict function looks like this:
```python
def predict(self, inputs):
	summation = np.dot(np.insert(inputs, 0, 1), self.weights)
	return self.activate(summation)
```

##### Training
Model training, in general, is the process of teaching a machine learning algorithm to make accurate predictions based on the input data. Our training function looks like so:
```python
def train(self, X, y):
```
where `X` is the input data and `y` represents the target labels.

 In the context of the single layer perceptron, the training loop consists of 3 main steps:

1. *Initialization*: The Perceptron starts with randomly initialized weights set earlier in the `__init__` function, and is not initialized with the bias term:
```python
X = np.insert(X, 0, 1, axis=1)  # Add bias term
```

2. *Training Loop:* The main training loop contains multiple steps, including:

- Iterating through epochs and create total error:
```python
for epoch in range(self.num_epochs):
 total_error = 0
```
- Processing each sample, iterating over each training example:
```python
for inputs, label in zip(X, y):
```
- Making predictions using our predictions function:
```python
prediction = self.predict(inputs[1:])
```
- Calculating errors by taking the difference between the true label and prediction:
```python
error = label - prediction
total_error += abs(error)
```
- Updating weights based on the error, learning rate, and input values:
```python
self.weights += self.learning_rate * error * inputs
```

The full train function looks like this:
```python
def train(self, X, y):
    X = np.insert(X, 0, 1, axis=1)  # Add bias term
    for epoch in range(self.num_epochs):
        total_error = 0
        for inputs, label in zip(X, y):
            prediction = self.predict(inputs[1:])
            error = label - prediction
            total_error += abs(error)
            self.weights += self.learning_rate * error * inputs

        # Print training progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: weights = {self.weights}, total error = {total_error}")

        # Early stopping if perfect classification
        if total_error == 0:
            print(f"Converged at epoch {epoch}")
            break
```

3. *Evaluation and update:* Evaluations and updates are not as relevant in a single-layer perceptron. Here, it is just printing information and checking for perfect classification, which is a simple form of early-stopping. But in a more complex neural network, this step would include things like:

- Backpropagation
- Gradient Descent
- More advanced early stopping
- Learning rate adjustment
- Regularization
- Normalization
- Checkpointing

```python
if epoch % 100 == 0:
            print(f"Epoch {epoch}: weights = {self.weights}, total error = {total_error}")
        
        if total_error == 0:
            print(f"Converged at epoch {epoch}")
            break
```

##### Usage
The Single-Layer Perceptron is a very limited architecture. It is difficult to find a problem that this model will excel at. To demonstrate some of the limitations of the model, we can look at the XOR problem.

The XOR problem, also referred to as the "exclusive or" problem, is the problem of using a neural network to predict the outputs of XOR logic gates given two binary inputs. It is a problem because the XOR function is not linearly seperable, meaning you can't draw a single straight line on the 2D plot that seperates 1s from 0s. 

Imagine plotting the following points on a 2D graph:

- (0,0) -> Output 0
- (0,1) -> Output 1
- (1,0) -> Output 1
- (1,1) -> Output 0

There is no way to draw a single straight line that seperates the 1s from 0s. This matters in the context of single-layer perceptrons because it tries to find a linear decision boundary (imagine a straight line on a 2D plot) to seperate classes. This is solvable by multi-layer neural networks.

Here is how the problem is set up in code:
```python
if __name__ == "__main__":
    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # AND problem
    y_and = np.array([0, 0, 0, 1])
    perceptron_and = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_and.train(X, y_and)
    
    print("\nAND function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_and.predict(inputs)}")

    # OR problem
    y_or = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_or.train(X, y_or)
    
    print("\nOR function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_or.predict(inputs)}")

    # XOR problem
    y_xor = np.array([0, 1, 1, 0])
    perceptron_xor = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_xor.train(X, y_xor)
    
    print("\nXOR function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_xor.predict(inputs)}")
```

This is the output:

```
Epoch 0: weights = [ 0.00359545  0.02398612 -0.45497649], total error = 2
Converged at epoch 11

AND function results:
Input: [0 0], Prediction: 0
Input: [0 1], Prediction: 0
Input: [1 0], Prediction: 0
Input: [1 1], Prediction: 1
Epoch 0: weights = [-2.52805087 -0.31402282 -0.41771306], total error = 3
Converged at epoch 8

OR function results:
Input: [0 0], Prediction: 0
Input: [0 1], Prediction: 1
Input: [1 0], Prediction: 1
Input: [1 1], Prediction: 1
Epoch 0: weights = [-0.46176641  1.40984421  0.78214246], total error = 1
Epoch 100: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 200: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 300: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 400: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 500: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 600: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 700: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 800: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4
Epoch 900: weights = [ 0.03823359 -0.09015579 -0.01785754], total error = 4

XOR function results:
Input: [0 0], Prediction: 1
Input: [0 1], Prediction: 1
Input: [1 0], Prediction: 0
Input: [1 1], Prediction: 0
```

The results clearly demonstrate the capabilities and limitations of single-layer perceptrons. The model successfully learned the AND and OR functions, converging quickly (at epochs 11 and 8 respectively) and making correct predictions for all inputs. However, it failed to solve the XOR problem. For XOR, the perceptron did not converge even after 1000 epochs, with the total error remaining constant at 4 from epoch 100 onwards. It made incorrect predictions for [0,0] and [1,0] inputs, illustrating its inability to create the non-linear decision boundary required for XOR classification. This outcome confirms that single-layer perceptrons cannot solve non-linearly separable problems like XOR, highlighting the need for more complex neural network architectures in such cases.

##### Full implementation
```python
import numpy as np

class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, num_epochs=1000):
        self.num_features = num_features
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.weights = np.random.randn(num_features + 1)  # Random initialization

    def activate(self, x):

        return 1 if x >= 0 else 0

    def predict(self, inputs):
        summation = np.dot(np.insert(inputs, 0, 1), self.weights)
        return self.activate(summation)

    def train(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        for epoch in range(self.num_epochs):
            total_error = 0
            for inputs, label in zip(X, y):
                prediction = self.predict(inputs[1:])
                error = label - prediction
                total_error += abs(error)
                self.weights += self.learning_rate * error * inputs

            # Print training progress
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: weights = {self.weights}, total error = {total_error}")

            # Early stopping if perfect classification
            if total_error == 0:
                print(f"Converged at epoch {epoch}")
                break


if __name__ == "__main__":
    # Input data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # AND problem
    y_and = np.array([0, 0, 0, 1])
    perceptron_and = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_and.train(X, y_and)

    print("\nAND function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_and.predict(inputs)}")

    # OR problem
    y_or = np.array([0, 1, 1, 1])
    perceptron_or = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_or.train(X, y_or)

    print("\nOR function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_or.predict(inputs)}")

    # XOR problem
    y_xor = np.array([0, 1, 1, 0])
    perceptron_xor = Perceptron(num_features=2, learning_rate=0.1, num_epochs=1000)
    perceptron_xor.train(X, y_xor)

    print("\nXOR function results:")
    for inputs in X:
        print(f"Input: {inputs}, Prediction: {perceptron_xor.predict(inputs)}")
```
