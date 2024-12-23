import numpy
import time
import random

class Network:
    def __init__(self, sizes: list[int]):
        self.sizes = sizes
        self.biases = [numpy.random.randn(layer_size, 1) for layer_size in sizes[1:]]
        self.weights = [numpy.random.randn(sizes[i+1], sizes[i]) for i in range(len(sizes)-1)]


    def train(self, training_data: list[tuple], epoches: int, mini_batch_size: int, learning_rate: float, test_data:list[tuple]):
        print("training...")
        for j in range(epoches):
            random.shuffle(training_data)
            start_time = time.time()
            for i in range(0, len(training_data), mini_batch_size):
                mini_batch = training_data[i:i+mini_batch_size]
                self.run_mini_batch(mini_batch, learning_rate)

            end_time = time.time()
            corrects = self.evaluate(test_data)
            print(f"epoch {j}: {corrects}/{len(test_data)}; {end_time-start_time} seconds")


    def evaluate(self, test_data) -> int:
        res = 0
        for x, y in test_data:
            res += self.evaluate_single(x, y)
        return res
        

    def evaluate_single(self, x, y) -> int:
        # x is 784 stuff
        # by the end, is 10 stuffs [0.21, 0.57, ...]
        # y is 10 stuffs too [0.0, 1.0, 0.0, ...] (2 is correct number)
        if numpy.argmax(self.feedforward(x)) == numpy.argmax(y):
            return 1 
        return 0


    def run_mini_batch(self, mini_batch: list, learning_rate: int):
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]  # zero-filled list of vectors to store change per mini batch
        nabla_w = [numpy.zeros(w.shape) for w in self.weights] # zero-filled list of matrices to store chang per mini batch
        for inputs_x, answer_y in mini_batch:
            # deltas are the change per single training example
            delta_nabla_w, delta_nabla_b = self.backprop(inputs_x, answer_y)

            # sum up the single changes for the whole mini batch
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]# nabla_b + delta_nabla_b
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] # nabla_w + delta_nabla_w
        self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)] # += nabla_w    # normalized to batch size and effceted by learning rate
        self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]  # += nabla_b
        # print("post mini", self.biases[1])


    # def feedforward(self, x):
    #     # place first layer activations
    #     # compute next layer (vectorized according to fancy equation on ur paper)
    #     # reach last layer and return vector of activations (or integer of highest rated activation)
    #     for b, w in zip(self.biases, self.weights):
    #         x = sigmoid(numpy.dot(w, x)+b) # happens vector wise
    #     return x
    

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(numpy.dot(w, a)+b)
        return a


    def backprop(self, x, y):
        # init zero filled biases and weights
        nabla_b = [numpy.zeros(b.shape) for b in self.biases]  
        nabla_w = [numpy.zeros(w.shape) for w in self.weights] 

        # feedforward
        # while feedforwarding store activations and z's
        activations = [x]
        zs = [None, ]
        for b, w in zip(self.biases, self.weights):
            z = numpy.dot(w, activations[-1])+b
            zs.append(z)
            # store z and activation for backprop   (z is pre-sigmoid activation)
            activation = sigmoid(z)
            activations.append(activation)
            
        # first, calculate error^L
        error =  (activations[-1] - y) * sigmoid_prime(zs[-1])
        nabla_w[-1] = numpy.dot(error, activations[-2].transpose())
        nabla_b[-1] = error

        # cost in respect to weights/biases
        # activation in layer which weights is coming out from, multipled by error in layer is going into (of course.. vector wise)
        # for biases, it's literally just the error in the layer

        # then, in loop, backpropgates for error^l
        for i in range(2, len(self.sizes)):
            # error = numpy.dot(self.weights[-i+1].transpose() * sigmoid_prime(zs[-i]), error)
            error = numpy.dot(self.weights[-i+1].transpose(), error) * sigmoid_prime(zs[-i])
            nabla_w[-i] = numpy.dot(error, activations[-i-1].transpose())
            nabla_b[-i] = error

        return nabla_w, nabla_b


# helpers
def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + numpy.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


import mnist_loader
def main():
    print('baba is not you!')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.train(training_data, 30, 10, 3, test_data)


if __name__ == "__main__":
    main()

        