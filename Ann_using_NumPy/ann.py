# import library:
from tkinter import E
import numpy as np

class ann(object):
    def __init__(self,data):
        self.data = data
        self.m = np.array(self.data).shape[0]

    #prepare the data:
    def train_test_split(self):
        """
        This function will split our data into train and test samples.
        """
        try:
            data = np.array(self.data)

            m,n = data.shape

            # Shuffle the data before splitting it:
            np.random.shuffle(data)

            #prepare test data:
            data_test = data[0:1000].T
            Y_test = data_test[0]
            X_test = data_test[1:n]
            X_test = X_test / 255.

            # Prepare the train data:
            data_train = data[1000:m].T
            Y_train = data_train[0]
            X_train = data_train[1:n]
            X_train = X_train / 255.

            return X_train,Y_train,X_test,Y_test
        except Exception as e:
            raise Exception

    # initialize the base parameters
    def __init_params(self):
        """
        This holds the initial parameters for the neural net.
        """
        try:
            W1 = np.random.rand(10,784) - 0.5
            b1 = np.random.rand(10,1) - 0.5
            W2 = np.random.rand(10,10) - 0.5
            b2 = np.random.rand(10,1) - 0.5
            return W1,b1,W2,b2
        except Exception as e:
            raise e

    # Relu Activation function
    def __ReLU(self,Z):
        """
        ReLU : Rectified linear unit

        Z = {
            0 if Z < 0,
            Z if Z > 0
        }
        """
        return np.maximum(Z,0)

    # Softmax activation function
    def __softmax(self,Z):
        """
        Softmax : used for multiclass classification.
        """
        A = np.exp(Z) / sum(np.exp(Z))
        return A

    # Derrivative of ReLU
    def __derriv_ReLU(self,Z):
        """
        Derriviative of ReLU.
        """
        return Z > 0
    
    # One hot encoding of labels:
    def __one_hot(self,Y):
        """
        One hot encoding of Y values.
        """
        one_hot_Y = np.zeros((Y.size,Y.max()+1))
        one_hot_Y[np.arange(Y.size), Y] = 1
        one_hot_Y = one_hot_Y.T
        return one_hot_Y


    # Forward propagation
    def forward_prop(self,W1,b1,W2,b2,X):
        """
        Forward propagation:
        weighted sum = dot product of initial weigths with input values.
        Z = (weighted sum of inputs) + bias
        A = ReLU(Z)
        """
        try:
            Z1 = W1.dot(X) + b1
            A1 = self.__ReLU(Z1)
            Z2 = W2.dot(A1) + b2
            A2 = self.__softmax(Z2)
            return Z1,A1,Z2,A2
        except Exception as e:
            raise e

    # Backward propagation
    def back_prop(self,Z1,A1,Z2,A2,W1,W2,X,Y):
        """
        Backward propagation:
        delta(W) = -(derriviative(total error)/derrivative(weights))
        ----------After derrivation trough the chain rule------------
        delta(W) = 1/(total data points) * derrivative(errors).(A.Transpose)
        """
        try:
            one_hot_Y = self.__one_hot(Y)
            dZ2 = A2 - one_hot_Y
            dW2 = 1 / self.m * dZ2.dot(A1.T)
            db2 = 1 / self.m * np.sum(dZ2)
            dZ1 = W2.T.dot(dZ2) * self.__derriv_ReLU(Z1)
            dW1 = 1 / self.m * dZ1.dot(X.T)
            db1 = 1 / self.m * np.sum(dZ1)
            return dW1,db1,dW2,db2
        except Exception:
            raise Exception
    # Update the weights
    def update_params(self,W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate):
        """
        new_params(w/b) = old_param - learning_rate(derrivate(old_weights))
        """
        try:
            W1 = W1 - learning_rate * dW1
            b1 = b1 - learning_rate * db1
            W2 = W2 - learning_rate * dW2
            b2 = b2 - learning_rate * db2
            return W1,b1,W2,b2
        except Exception:
            raise Exception
    
    # Prediction:
    def get_prediction(self,A2):
        return np.argmax(A2,0)
    
    # Accuracy:
    def get_accuracy(self,predictions,Y):
        return np.sum(predictions == Y) / Y.size
    
    # Gradient descent
    def fit(self,X,Y,learning_rate,epochs,verbose=False):
        """
        This will train our neural net.
        """
        try:
            # initial weights and biases:
            W1,b1,W2,b2 = self.__init_params()
            for i in range(epochs):
                #forward propagation 
                Z1,A1,Z2,A2 = self.forward_prop(W1,b1,W2,b2,X)
                # back prop:
                dW1,db1,dW2,db2 = self.back_prop(Z1,A1,Z2,A2,W1,W2,X,Y)
                # update the parameters
                W1,b1,W2,b2 = self.update_params(W1,b1,W2,b2,dW1,db1,dW2,db2,learning_rate)
                if verbose == True:
                    if i % 20 == 0: # to print after every 20th epoch
                        print("Iteration :: ",i)
                        predictions = self.get_prediction(A2)
                        print(f"Accuracy :: ",self.get_accuracy(predictions,Y))

            return W1,b1,W2,b2
        except Exception:
            raise Exception
    
    # to make predictions:
    def make_predictions(self,X,W1,b1,W2,b2):
        """
        This will make some prediction on the unseen data.
        """
        try:
            _,_,_,A2 = self.forward_prop(W1,b1,W2,b2,X)
            predictions = self.get_prediction(A2)
            return predictions
        except Exception:
            raise Exception
