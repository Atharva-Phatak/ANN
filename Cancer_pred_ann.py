#USE OF ARTIFICAL NEURAL NETS TO PREDICT BREAST CANCER

#IMPORTIMG THE LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#GETTING THE DATA
dataset = pd.read_csv('B_cancer.csv')

#GETTING THE DEPENDENT VARIABLE
y = dataset.iloc[:,1].values
#GETTING THE INDEPENDENT VARIABLES
X = dataset.iloc[:,2:12].values

#PREPROCESSING OF THE DATA

#ENCODING THE DATA
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#CREATING TRAINING AND TEST DATA
from sklearn.model_selection import train_test_split

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.2, random_state = 0)

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#IMPORTING THE KERAS LIBRARIES
from keras.models import Sequential
from keras.layers import Dense

#INITIALISING THE ANN
classifier = Sequential()

#ADDING LAYERS TO THE NEURAL NETS
classifier.add(Dense(units = 5 , kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 10))

classifier.add(Dense(units = 5 , kernel_initializer = 'uniform' , activation = 'relu'))

#ADDING THE OUTPUT LAYER
classifier.add(Dense(units = 1 , kernel_initializer = 'uniform' , activation = 'sigmoid'))

#COMPILING THE ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

#FIITNG THE ANN TO THE TRAINING SET
classifier.fit(X_train , y_train ,batch_size = 10, nb_epoch = 100)

#MAKING THE PREDICITIONS
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.8)

#MAKING THE CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)

#CHECKING THE ACCURACY
from sklearn.metrics import accuracy_score
print("THE ACCURACY OF THE MODEL IN PERCENT IS: {}".format(accuracy_score(y_test , y_pred) *100))










