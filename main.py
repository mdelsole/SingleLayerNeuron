import numpy as np
np.random.seed(1)
x = np.array([[1,0,1], [0,1,0], [1,1,1], [1,0,0]])
y = np.array([[1,0,1,0]]).T

print("Random starting weights: ")
synaptic_weights = np.random.random((3,1))
print(synaptic_weights)

for iteration in range(1000):
    z = np.dot(x, synaptic_weights)
    sigmoid = 1/(1+np.exp(-z))
    error = (y - sigmoid)
    sigmoidDerivative = sigmoid * (1 - sigmoid)
    # How can one adjustment apply to all weights?
    adjustment = np.dot(x.T, error*sigmoidDerivative)
    synaptic_weights += adjustment

print("New synaptic weights after training: ")
print(synaptic_weights)

print("Considering new situation: [0,1,1]")
newZ = np.dot(np.array([0,1,1]), synaptic_weights)
newSigmoid = 1/(1+np.exp(-newZ))
print(newSigmoid)


