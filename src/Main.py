from ActivationFunction.Step import step
from ActivationFunction.Sigmoid import sigmoid
from ActivationFunction.LeakyReLU import leakyrelu

x = -10
print(step(x))

sig = sigmoid(x)
output = sig.forward()
print(output)
gard = 0.2
error = sig.backword(gard)

x = -0.1
print(leakyrelu(x))