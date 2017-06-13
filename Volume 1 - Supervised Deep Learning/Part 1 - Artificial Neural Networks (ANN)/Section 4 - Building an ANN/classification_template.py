'''

import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

def get_session(gpu_fraction=.4):

    num_threads = os.environ.get('OMP_NUM_THREADS')
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

    if num_threads:
        print("hi")
        return tf.Session(config=tf.ConfigProto(
            gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
    else:
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())
'''
# Classification template

# Importing the libraries
import numpy as np
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
label_encoder_X_1 = LabelEncoder()
X[:, 1] = label_encoder_X_1.fit_transform(X[:,1])
label_encoder_X_2 = LabelEncoder()
X[:, 2] = label_encoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



# Fitting classifier to the Training set
# Create your classifier here

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout

'''
classifier = Sequential()
classifier.add(Dense(units = 6, input_dim = 11))
classifier.add(Activation("relu"))
classifier.add(Dense(units = 6))
classifier.add(Activation("relu"))
classifier.add(Dense(units = 1))
classifier.add(Activation("sigmoid"))

classifier.compile(optimizer="adam",loss = "binary_crossentropy", metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)

y_pred = (classifier.predict(X_test) >= .5)
'''

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, input_dim = 11))
    classifier.add(Activation("relu"))
    classifier.add(Dense(units = 6))
    classifier.add(Activation("relu"))
    classifier.add(Dense(units = 1))
    classifier.add(Activation("sigmoid"))
    classifier.compile(optimizer="adam",loss = "binary_crossentropy", metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, nb_epoch = 100 )
accuracies = cross_val_score(estimator=classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, input_dim = 11))
    classifier.add(Activation("relu"))
    classifier.add(Dense(units = 6))
    classifier.add(Activation("relu"))
    classifier.add(Dense(units = 1))
    classifier.add(Activation("sigmoid"))
    classifier.compile(optimizer=optimizer,loss = "binary_crossentropy", metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size':[25,32],
                'nb_epoch':[100,500],
                'optimizer':['adam','rmsprop'],}
grid_search = GridSearchCV(estimator=classifier, param_grid = parameters, scoring='accuracy', cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

