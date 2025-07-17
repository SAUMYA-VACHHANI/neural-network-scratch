import numpy as np

class SimpleLinearModel:
    def __init__(self):
        self.weight = None
        self.bias = None

    def initialize_weights(self, n_inputs):
        np.random.seed(42)
        self.weight = np.random.randn(n_inputs, 1)  # Shape: (n_inputs, 1)
        self.bias = np.random.randn(1)
        return self.weight, self.bias

    def forward(self, inputs):
        return np.dot(inputs, self.weight) + self.bias

    def backward_propagation(self, learning_rate, inputs, targets):
        predictions = self.forward(inputs)
        error = predictions - targets
        n_samples = inputs.shape[0]
        self.weight_gradient = (1 / n_samples) * np.dot(inputs.T, error)
        self.bias_gradient = (1 / n_samples) * np.sum(error)
        self.weight -= learning_rate * self.weight_gradient
        self.bias -= learning_rate * self.bias_gradient
        return self.weight, self.bias

    def fit(self, inputs, targets, learning_rate, epochs):
        self.initialize_weights(inputs.shape[1])
        for _ in range(epochs):
            self.forward(inputs)
            self.backward_propagation(learning_rate, inputs, targets)
        return self

    def predict(self, inputs):
        return self.forward(inputs)

if __name__ == "__main__":
    inputs = np.array([[0,1,1],[1,1,2],[1,2,3],[2,3,5],[3,5,8],[5,8,13]])
    targets = np.array([[2],[3],[5],[8],[13],[21]])
    model = SimpleLinearModel()
    model.fit(inputs=inputs, targets=targets, learning_rate=0.01, epochs=1000)
    num1 = int(input("Enter the first number: "))
    num2 = int(input("Enter the second number: "))
    num3 = int(input("Enter the third number: "))
    print(np.ceil(model.predict([num1, num2, num3])))