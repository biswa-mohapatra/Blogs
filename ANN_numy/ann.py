import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class artificial_neural_net(object):
    """
    Artificaial Neural network.
    """
    def __init__(self,data):
        self.data = data
        self.m = np.array(self.data).shape[0]
    def train_test_split(self):
        """
        Load the data and return x_test,y_test,x_train and y_train parameters.
        """
        try:
            load_data = np.array(self.data)
            data = np.array(load_data)
            m, n = data.shape
            # shuffle before splitting into dev and training sets
            np.random.shuffle(load_data)

            # prepare the dev/test data:
            data_test = load_data[0:1000].T
            Y_test = data_test[0]
            X_test = data_test[1:n]
            X_test = X_test / 255. 

            # prepare the training:
            data_train = load_data[1000:m].T
            Y_train = data_train[0]
            X_train = data_train[1:n]
            X_train = X_train / 255.
            _,m_train = X_train.shape

            return X_train, Y_train, X_test, Y_test
        except ValueError:
            raise ValueError
        except Exception:
            raise Exception
    
    # initialize the base parameters for neural nets.
    def __init_params(self):
        """
        Initializing the base parameters for neural networks.
        """
        try:
            W1 = np.random.rand(10, 784) - 0.5
            b1 = np.random.rand(10, 1) - 0.5
            W2 = np.random.rand(10, 10) - 0.5
            b2 = np.random.rand(10, 1) - 0.5
            return W1, b1, W2, b2
        except Exception:
            raise Exception

    def __ReLU(self,Z):
        """
        Relu : Rectified linear unit.
        Z = {
            0 if Z < 0,
            Z if Z > 0
        }
        """
        return np.maximum(Z,0)
    
    def __softmax(self,Z):
        """
        Softmax : Multiclass classification problems.
        Sigmoid : Binary classification problems.

        softmax = ratio between exponantial z value and summation of exponantial z values.
        """
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    def __derriv_ReLU(self,Z):
        return Z > 0

    def __one_hot(self,Y):
        """
        One-hot : It will change the labels to 1 if it the correct position of the label
        and sets everything else to 0

        Example:
        array = [A,B,C,D]
        one-hot(array) =        A B C D
                            A   1 0 0 0
                            B   0 1 0 0
                            C   0 0 1 0
                            D   0 0 0 1

        """
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y

    def forward_prop(self,W1, b1, W2, b2, X):
        """
        Forward propagation.
        weighted sum = dot product of initial weights with input value.
        Z = (weighted sum of inputs) + bias term
        A = ReLU(Z)
        """
        try:
            Z1 = W1.dot(X) + b1
            A1 = self.__ReLU(Z1)
            Z2 = W2.dot(A1) + b2
            A2 = self.__softmax(Z2)
            return Z1, A1, Z2, A2
        except Exception:
            raise Exception

    def back_prop(self,Z1, A1, Z2, A2, W1, W2, X, Y):
        """
        Backward propagation.
        delta(w) = -(derv(error total)/derv(weight))
        ----------After derrivation trough chain rule-----------
        delta(w) = 1/(total rows of data)*derv(error).(transpose A) 
        """
        try:
            one_hot_Y = self.__one_hot(Y)
            dZ2 = A2 - one_hot_Y
            dW2 = 1 / self.m * dZ2.dot(A1.T)
            db2 = 1 / self.m * np.sum(dZ2)
            dZ1 = W2.T.dot(dZ2) * self.__derriv_ReLU(Z1)
            dW1 = 1 / self.m * dZ1.dot(X.T)
            db1 = 1 / self.m * np.sum(dZ1)
            return dW1, db1, dW2, db2
        except Exception:
            raise Exception

    def update_params(self,W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
        """
        new_param(w/b) = old_param - lr(derv(old_weight))
        """
        try:
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1    
            W2 = W2 - learning_rate * dW2  
            b2 = b2 - learning_rate * db2    
            return W1, b1, W2, b2
        except Exception:
            raise Exception

    def get_predictions(self,A2):
        return np.argmax(A2, 0)

    def get_accuracy(self,predictions, Y):
        #print(predictions, Y)
        return np.sum(predictions == Y) / Y.size


    def fit(self,X,Y,learning_rate,epochs,verbose=False):
        """
        This will train the model.
        """
        try:
            W1, b1, W2, b2 = self.__init_params()
            for i in range(epochs):
                Z1, A1, Z2, A2 = self.forward_prop(W1, b1, W2, b2, X)
                dW1, db1, dW2, db2 = self.back_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
                W1, b1, W2, b2 = self.update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
                if verbose == True:
                    if i % 10 == 0:
                        print("Iteration: ", i)
                        predictions = self.get_predictions(A2)
                        print(self.get_accuracy(predictions, Y))
            return W1, b1, W2, b2
        except Exception:
            raise Exception
    
    def make_predictions(self,X, W1, b1, W2, b2):
        """
        Will make predictions for the given unseen data.
        """
        try:
            _, _, _, A2 = self.forward_prop(W1, b1, W2, b2, X)
            #A2 = np.array(A2)
            #print(A2.flatten())
            predictions = self.get_predictions(A2)
            return predictions
        except Exception:
            raise Exception
