# -*- coding: utf-8 -*-
"""
Trains a convolutional neural network on the CIFAR-10 dataset, then generated adversarial images using the
PGD/BIM attack and retrains the network on the training set augmented with the adversarial images.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging

import keras.backend as k
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, Dropout
import numpy as np
from art.attacks.iterative_method import BasicIterativeMethod
from art.classifiers import KerasClassifier
from art.utils import load_dataset
from datetime import datetime
import os 

print('retraining with pgd attack')
# Configure a logger to capture ART outputs; these are printed in console and the level of detail is set to INFO
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Read CIFAR10 dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('cifar10'))
x_train, y_train = x_train[:5000], y_train[:5000]
x_test, y_test = x_test[:500], y_test[:500]
im_shape = x_train[0].shape

# Create Keras convolutional neural network - basic architecture from Keras examples
# Source here: https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
k.set_learning_phase(1)
model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Create classifier wrapper
classifier = KerasClassifier(clip_values=(0, 1), model=model)
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

# Craft adversarial samples with BIM
logger.info('Create BIM attack')
adv_crafter = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.01, max_iter=40)
logger.info('Craft attack on training examples')
x_train_adv = adv_crafter.generate(x_train)
logger.info('Craft attack test examples')
x_test_adv = adv_crafter.generate(x_test)

# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Classifier before adversarial training')
logger.info('Accuracy on adversarial samples: %.2f%%', (acc * 100))

# Data augmentation: expand the training set with the adversarial samples
x_train = np.append(x_train, x_train_adv, axis=0)
y_train = np.append(y_train, y_train, axis=0)

# Retrain the CNN on the extended dataset
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(x_train, y_train, nb_epochs=10, batch_size=128)

# Evaluate the adversarially trained classifier on the test set
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
logger.info('Classifier with adversarial training')
logger.info('Accuracy on adversarial samples: %.2f%%', (acc * 100))
print('adversarial-robustness-toolbox/examples/cifar_adversarial_training_from_ART_BIM.py')
#save the model
model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + 'cifar_adversarial_training_BIM.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_name)
print('Saved trained model at %s ' % model_path)
classifier.save(model_path)
