"""
Modified version of ART mnist_transferability under examples folder
 
1. Generate adversarial images by performing various adversarial attack techniques on source classifier
2. Use them to attack an adversarially trained model(from ART) and original model. 
This is to show effect of transfer attack.
"""
from __future__ import absolute_import, division, print_function

import sys
from os.path import abspath

sys.path.append(abspath('.'))
from keras.models import Sequential, Model
import numpy as np
from keras.layers import Input, Dense, Flatten
from art.utils import load_mnist
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as pgd
from art.attacks.carlini import CarliniLInfMethod as cw
from art.attacks.fast_gradient import FastGradientMethod as fgsm
from art.attacks.deepfool import DeepFool
import pickle
# Load model
from keras.models import load_model
from art.classifiers import KerasClassifier
import random
random.seed(1)
# dataset
with open('adversarial-robustness-toolbox/dataset/x_train.pickle', 'rb') as infile:
    x_train = pickle.load(infile)
with open('adversarial-robustness-toolbox/dataset/y_train.pickle', 'rb') as infile:
    y_train = pickle.load(infile)
with open('adversarial-robustness-toolbox/dataset/x_test.pickle', 'rb') as infile:
    x_test = pickle.load(infile)
with open('adversarial-robustness-toolbox/dataset/y_test.pickle', 'rb') as infile:
    y_test = pickle.load(infile)
# Source classifier: accuracy =0.88
robust_classifier = load_model('saved_models/cifar10_original.h5')
source = KerasClassifier(clip_values=(0, 1), model=robust_classifier, use_logits=False)

# Compare with existing Adversarial Training (from ART)
robust_classifier = load_model('saved_models/cifar_adversarial_training_DF.h5')
robust_classifier = KerasClassifier(clip_values=(0, 1), model=robust_classifier, use_logits=False)

# Normal images
original_model = load_model('saved_models/cifar_resnet.h5')  # original
classifier = KerasClassifier(clip_values=(0, 1), model=original_model, use_logits=False)

# Under DeepFool attack
true_label = np.argmax(y_test[:100], axis=1)
adv_crafter = DeepFool(source)
adv_df = adv_crafter.generate(x_test[0:100])
preds = np.argmax(classifier.predict(adv_df), axis=1)
acc_df = np.sum(preds == true_label) 
print("Test accuracy for original classifier under DeepFool (transfered attack): %.1f%%" % (acc_df))
#FGSM
attacker_fgsm = fgsm(source, eps=0.3)
print('maximum perturbation for fgsm and pgd are 0.3')
adv_fgsm = attacker_fgsm.generate(x_test[:100])
pred_fgsm = np.argmax(classifier.predict(adv_fgsm), axis=1)
acc_fgsm = np.sum(pred_fgsm == true_label)
print("Test accuracy for original classifier under FGSM (transfered attack): %.1f%%" % (acc_fgsm))
# PGD
attacker_pgd = pgd(source, eps=0.3)
adv_pgd = attacker_pgd.generate(x_test[:100])
pred_pgd = np.argmax(classifier.predict(adv_pgd), axis=1)
acc_pgd = np.sum(pred_pgd == true_label)
print("Test accuracy for original classifier under PGD (transfered attack): %.1f%%" % (acc_pgd))

# CW
attacker = cw(source, targeted=False, batch_size=100)
adv_cw = attacker.generate(x_test[:100])
pred_cw = np.argmax(classifier.predict(adv_cw), axis=1)
acc_cw = np.sum(pred_cw == true_label)
print("Test accuracy for original classifier under CW (transfered attack): %.1f%%" % (acc_cw))
##########################################################################################################################
'''Adversarially trained'''
#FGSM
robust_pred = np.argmax(robust_classifier.predict(adv_fgsm), axis=1)
nb_correct_adv_robust_pred = np.sum(robust_pred == true_label)
print("Correctly classified by adversarially retrained model (ART) against FGSM attack (transfered attack): {}".format(nb_correct_adv_robust_pred))
# PGD
robust_pred_pgd = np.argmax(robust_classifier.predict(adv_pgd), axis=1)
nb_correct_adv_robust_pred = np.sum(robust_pred_pgd == true_label)
print("Correctly classified against PGD attack (transfered attack): {}".format(nb_correct_adv_robust_pred))
# CW
robust_pred_cw = np.argmax(robust_classifier.predict(adv_cw), axis=1)
nb_correct_adv_robust_pred = np.sum(robust_pred_cw == true_label)
print("Correctly classified against CW attack (transfered attack): {}".format(nb_correct_adv_robust_pred))

# Craft adversarial samples with DeepFool
robust_pred_df = np.argmax(robust_classifier.predict(adv_df), axis=1)
nb_correct_adv_robust_pred_df = np.sum(robust_pred_df == true_label)
print("Correctly classified against DeepFool attack (transfered attack): {}".format(nb_correct_adv_robust_pred_df))
print('8/28')
print('cifar_transferability.py for cifar-10 dataset')
print('based on inf norm')