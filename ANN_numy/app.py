import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ann import artificial_neural_net

## Preparing the data
data = pd.read_csv("fashion-mnist_train.csv/fashion-mnist_train.csv")

# Defining data:
f_mnist_data = {0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'}

# Calling the artificial neural network and passing the data
ann = artificial_neural_net(data)

X_train, Y_train, X_test, Y_test = ann.train_test_split()

W1, b1, W2, b2 = ann.fit(X_train,Y_train,learning_rate=0.15,epochs=10,verbose=True)

index = 0

prediction = ann.make_predictions(X_train[:,index,None],W1, b1, W2, b2)

def test_prediction(prediction,index):
    current_image = X_train[:, index, None]
    label = Y_train[index]
    prediction = int(prediction)
    print("Prediction: ", f_mnist_data[prediction])
    print("Label: ", f_mnist_data[label])
    
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

test_prediction(prediction,index)

dev_predictions = ann.make_predictions(X_test, W1, b1, W2, b2)
accuracy = ann.get_accuracy(dev_predictions, Y_test)

print(accuracy)