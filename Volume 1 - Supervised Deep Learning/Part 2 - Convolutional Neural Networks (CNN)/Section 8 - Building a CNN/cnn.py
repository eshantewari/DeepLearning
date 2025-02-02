# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 4000,
                         epochs = 10,
                         validation_data = test_set,
                         validation_steps = 2000)


from keras.models import load_model
#classifier.save('cat_dog.h5')
classifier = load_model('cat_dog.h5')

import numpy as np
from keras.preprocessing import image
test_image_dumbo = image.load_img('dataset/single_prediction/dumbo.JPG',target_size=(64,64))
test_image_dumbo1 = image.load_img('dataset/single_prediction/dumbo1.JPG',target_size=(64,64))
test_image_dumbo = np.expand_dims(image.img_to_array(test_image_dumbo),axis=0)
test_image_dumbo1 = np.expand_dims(image.img_to_array(test_image_dumbo1),axis=0)
result_dumbo = classifier.predict(test_image_dumbo)
result_dumbo1 = classifier.predict(test_image_dumbo1)
training_set.class_indices
def cat_or_dog(result):
    if result == 1:
        return 'dog'
    else:
        return 'cat'

prediction_dumbo = cat_or_dog(result_dumbo[0][0])
prediction_dumbo1 = cat_or_dog(result_dumbo1[0][0])