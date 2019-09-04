"""
Modified version of ART mnist_transferability under examples folder

Trains a CNN on the MNIST dataset using the Keras backend, then generates adversarial images using DeepFool
and uses them to attack a adversarial trained model(from ART) and original model. 
This is to show effect of transfer attack.
"""
from __future__ import absolute_import, division, print_function

import sys
from os.path import abspath

sys.path.append(abspath('.'))
from parallel_model import make_model_parallel
import keras
from keras.models import Sequential, Model
import numpy as np
from keras.layers import Input, Dense, Flatten
from art.utils import load_mnist
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as pgd
from art.attacks.carlini import CarliniLInfMethod as cw
from art.attacks.fast_gradient import FastGradientMethod as fgsm
from art.attacks.deepfool import DeepFool
# Load model
from keras.models import load_model
from art.classifiers import KerasClassifier
from keras.optimizers import Adam

def cnn_mnist_k():
    model =Sequential([
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')])

    image = Input(shape=(28, 28, 1))
    output = model(image)
    make_model_parallel(model)
    return Model(image, output)

# Read MNIST dataset
(x_train, y_train), (x_test, y_test), min_, max_ = load_mnist()

# Construct and train a convolutional neural network on MNIST using Keras
source = cnn_mnist_k()
source.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr = 0.01), 
                              metrics=['accuracy'])
source = KerasClassifier(clip_values=(min_, max_), model=source, use_logits=False)
source.fit(x_train, y_train, nb_epochs=5, batch_size=128)

# Craft adversarial samples with DeepFool
adv_crafter = DeepFool(source)
x_test_adv = adv_crafter.generate(x_test)

# Compare with existing Adversarial Training (from ART)
robust_classifier = load_model('saved_models/mnist_cnn_robust.h5')
robust_classifier = KerasClassifier(clip_values=(0, 1), model=robust_classifier, use_logits=False)
print('compare_transfer.py for mnist dataset v2')
print('based on inf norm')

# Normal images
original_model = load_model('saved_models/mnist_cnn_original.h5')  # original
classifier = KerasClassifier(clip_values=(0, 1), model=original_model, use_logits=False)

# Under DeepFool attack
true_label = np.argmax(y_test[:100], axis=1)
adv_crafter = DeepFool(source)
adv_df = adv_crafter.generate(x_test[0:100])
preds = np.argmax(classifier.predict(adv_df), axis=1)
acc_df = np.sum(preds == true_label) 
print("Test accuracy for original classifier under DeepFool (transfered attack): %.1f%%" % (acc_df))
#FGSM
attacker_robust_fgsm = fgsm(source, eps=0.3)
adv_fgsm = attacker_robust_fgsm.generate(x_test[:100])
x_test_adv_robust_pred_fgsm = np.argmax(classifier.predict(adv_fgsm), axis=1)
acc_fgsm = np.sum(x_test_adv_robust_pred_fgsm == true_label)
print("Test accuracy for original classifier under FGSM (transfered attack): %.1f%%" % (acc_fgsm))
# PGD
attacker_robust_pgd = pgd(source, eps=0.3)
adv_pgd = attacker_robust_pgd.generate(x_test[:100])
x_test_adv_robust_pred_pgd = np.argmax(classifier.predict(adv_pgd), axis=1)
acc_pgd = np.sum(x_test_adv_robust_pred_pgd == true_label)
print("Test accuracy for original classifier under PGD (transfered attack): %.1f%%" % (acc_pgd))

# CW
attacker = cw(source, targeted=False, batch_size=100)
adv_cw = attacker.generate(x_test[:100])
x_test_adv_pred = np.argmax(classifier.predict(adv_cw), axis=1)
acc_cw = np.sum(x_test_adv_pred == true_label)
print("Test accuracy for original classifier under CW (transfered attack): %.1f%%" % (acc_cw))
##########################################################################################################################
'''Adversarially trained'''
#FGSM
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(adv_fgsm), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified by adversarially retrained model (ART) against FGSM attack (transfered attack): {}".format(nb_correct_adv_robust_pred))
# PGD
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(adv_pgd), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified against PGD attack (transfered attack): {}".format(nb_correct_adv_robust_pred))
# CW
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(adv_cw), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified against CW attack (transfered attack): {}".format(nb_correct_adv_robust_pred))

# Craft adversarial samples with DeepFool
x_test_adv_robust_pred_df = np.argmax(robust_classifier.predict(adv_df), axis=1)
nb_correct_adv_robust_pred_df = np.sum(x_test_adv_robust_pred_df == true_label)
print("Correctly classified against DeepFool attack (transfered attack): {}".format(nb_correct_adv_robust_pred_df))
