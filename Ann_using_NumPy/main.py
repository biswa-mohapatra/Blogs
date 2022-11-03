# import some libraries
import numpy as np
import pandas as pd
from ann import ann

# define the data here:
f_mnist_data = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
    }
# prepare the data:
data = pd.read_csv("fashion-mnist_train.csv/fashion-mnist_train.csv")

# initialize the ann class here:
ann = ann(data)

# split the data to train and test:
X_train,Y_train,X_test,Y_test = ann.train_test_split()

# train our ann here:
W1,b1,W2,b2 = ann.fit(X_train,Y_train,learning_rate=0.15,epochs=750,verbose=True)

# make some predictions:
label = 0
prediction = int(ann.make_predictions(X_train[:,label,None],W1,b1,W2,b2))

print("Prediction: ", f_mnist_data[prediction])
print("Actual: ",f_mnist_data[label])

## Test the trained ann with unseen dataset:
test_pred = ann.make_predictions(X_test,W1,b1,W2,b2)
accuracy = ann.get_accuracy(test_pred,Y_test)

print("Accuracy on unseen dataset : ",accuracy)