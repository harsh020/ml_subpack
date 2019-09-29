from scipy import io
import numpy as np
from ..utils.dispImgs import disp_imgs
from ..neural_network import NeuralNetwork
from ..utils.utility import test_train_split

if __name__ == '__main__':
    mnist = io.loadmat('mnist_data/mnist')

    # print(mnist.keys())
    print(mnist['X'].shape)
    print(mnist['y'].shape)

    X = np.array(mnist['X'])
    y = np.array(mnist['y'])
    z = y.ravel()
    z = list(map(lambda x: 0 if x==10 else x, z))
    y = np.array(z).reshape(y.shape)

    # disp_imgs(X, 100, X.shape[1])

    # optional I guess
    for i in range(X.shape[0]):
        X_ = np.array(X[i, :]).reshape(20, 20)
        X_ = X_.T
        X[i, :] = list(X_.ravel())

    X_train, y_train, X_test, y_test = test_train_split(X, y, 0.2)

    # X = np.concatenate((np.ones(5000, 1), X), axis=1)

    # theta = io.loadmat('mnist_data/weights.mat')
    # theta_ = np.array(list(np.array(theta['Theta1']).ravel())+list(np.array(theta['Theta2']).ravel()))
    nn = NeuralNetwork(2, 50)
    nn.fit(X_train, y_train)
    predict = nn.predict(X_test)

    correct = 0
    for i in range(X_test.shape[0]):
        y_ = int(predict[i][0])
        y = int(y_test[i][0])
        if y_ == y:
            correct += 1

    print('\nPrediction Accuracy: {}\n' .format(round(correct/X_test.shape[0] * 100, 2)))


    more = 'y'
    while more == 'y':
        i = np.random.randint(X_test.shape[0])
        print('Predicted: {} | True: {}' .format(int(predict[i][0]), y_test[i][0]))
        more = input('Predict more <y/n>: ')

    # print(cost)
