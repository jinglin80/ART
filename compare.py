import numpy as np
from parallel_model import make_model_parallel
import pickle
import keras.backend as k
from keras.applications import vgg16
from keras.preprocessing import image
from keras.utils import np_utils
from art.defences import FeatureSqueezing
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as pgd
from art.attacks.carlini import CarliniL2Method as cw
from art.attacks.fast_gradient import FastGradientMethod as fgsm
from art.attacks.iterative_method import BasicIterativeMethod
from art.defences.adversarial_trainer import AdversarialTrainer
from art.attacks.deepfool import DeepFool
from datetime import datetime
# Load model
from keras.models import load_model
from art.classifiers import KerasClassifier
# dataset
with open('x_train_mnist.pickle', 'rb') as infile:
    x_train = pickle.load(infile)
with open('y_train_mnist.pickle', 'rb') as infile:
    y_train = pickle.load(infile)
with open('x_test_mnist.pickle', 'rb') as infile:
    x_test = pickle.load(infile)
with open('y_test_mnist.pickle', 'rb') as infile:
    y_test = pickle.load(infile)
###############################################################################################
# Compare with existing Adversarial Training (from ART)
robust_classifier = load_model('saved_models/mnist_cnn_robust.h5')
robust_classifier = KerasClassifier(clip_values=(0, 1), model=robust_classifier, use_logits=False)
# above model is trained use following code (from ART)
#attacks = BasicIterativeMethod(robust_classifier, eps=epsilon, eps_step=0.01, max_iter=40)
#trainer = AdversarialTrainer(robust_classifier, attacks, ratio=1.0)
#trainer.fit(x_train, y_train, nb_epochs=83, batch_size=50)
print('compare.py inf: 8/27')
epsilon = 0.1 # Maximum perturbation
print('eps = ', epsilon)
print('white-box attack on MNIST')
true_label = np.argmax(y_test[:100], axis=1)
pred = np.argmax(robust_classifier.predict(x_test[:100]), axis=1)
acc_orig = np.sum(pred == true_label)
print("Number of natural instances correctly classified by adversarially retrained model (ART) out of 100: {}".format(acc_orig))

#FGSM
attacker_robust = fgsm(robust_classifier, eps=epsilon)
x_test_adv_robust = attacker_robust.generate(x_test[:100])
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified by adversarially retrained model (ART) against FGSM attack: {}".format(nb_correct_adv_robust_pred))
# PGD
attacker_robust = pgd(robust_classifier, eps=epsilon)
x_test_adv_robust = attacker_robust.generate(x_test[:100])
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified against PGD attack: {}".format(nb_correct_adv_robust_pred))
# CW
attacker_robust = cw(robust_classifier, targeted=False, batch_size=100)
x_test_adv_robust = attacker_robust.generate(x_test[:100])
x_test_adv_robust_pred = np.argmax(robust_classifier.predict(x_test_adv_robust), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred == true_label)
print("Correctly classified against CW attack: {}".format(nb_correct_adv_robust_pred))
# DeepFool
adv_crafter_df = DeepFool(robust_classifier)
img_adv_df = adv_crafter_df.generate(x_test[0:100])
x_test_adv_robust_pred_df = np.argmax(robust_classifier.predict(img_adv_df), axis=1)
nb_correct_adv_robust_pred_df = np.sum(x_test_adv_robust_pred_df == true_label)
print("Correctly classified against DeepFool attack: {}".format(nb_correct_adv_robust_pred_df))

# Normal images
original_model = load_model('saved_models/mnist_cnn_original.h5')  # original
classifier = KerasClassifier(clip_values=(0, 1), model=original_model, use_logits=False)
x_test_pred = np.argmax(classifier.predict(x_test), axis=1)
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test, axis=1))/y_test.shape[0]*100
print("Test accuracy for normal instances: %.1f%%" % (nb_correct_pred))
# Under DeepFool attack
true_label = np.argmax(y_test[:100], axis=1)
adv_crafter = DeepFool(classifier)
img_adv_df = adv_crafter.generate(x_test[0:100])
preds = np.argmax(classifier.predict(img_adv_df), axis=1)
acc = np.sum(preds == true_label) 
print("Test accuracy for original classifier under DeepFool  : %.1f%%" % (acc))
#FGSM
attacker_robust_fgsm = fgsm(classifier, eps=epsilon)
x_test_adv_robust_fgsm = attacker_robust_fgsm.generate(x_test[:100])
x_test_adv_robust_pred_fgsm = np.argmax(classifier.predict(x_test_adv_robust_fgsm), axis=1)
nb_correct_adv_robust_pred_fgsm = np.sum(x_test_adv_robust_pred_fgsm == true_label)
print("Correctly classified against FGSM attack for original classifier: {}".format(nb_correct_adv_robust_pred_fgsm))
# PGD
# =============================================================================
# check 
# =============================================================================
attacker_robust_pgd = pgd(classifier, eps=epsilon)
x_test_adv_robust_pgd = attacker_robust_pgd.generate(x_test[:100])
x_test_adv_robust_pred_pgd = np.argmax(classifier.predict(x_test_adv_robust_pgd), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_robust_pred_pgd == true_label)
print("Correctly classified against PGD attack for original classifier: {}".format(nb_correct_adv_robust_pred))
# CW
attacker = cw(classifier, targeted=False, batch_size=100)
x_test_adv = attacker.generate(x_test[:100])
x_test_adv_pred = np.argmax(classifier.predict(x_test_adv), axis=1)
nb_correct_adv_robust_pred = np.sum(x_test_adv_pred == np.argmax(y_test[:100], axis=1))
print("Correctly classified against CW attack: {}".format(nb_correct_adv_robust_pred))
#######################
# Craft adversarial samples with DeepFool for our proposed model
#final_model = load_model('saved_models/92_mnist.h5')
#final_classifier = KerasClassifier(clip_values=(0,1), model=final_model, use_logits=False)
#adv_crafter_df_proposed = DeepFool(final_classifier)
#img_adv_df_proposed = adv_crafter_df_proposed.generate(x_test[0:100])
#preds_df = np.argmax(final_classifier.predict(img_adv_df_proposed), axis=1)
#acc = np.sum(preds_df == true_label) 
#print("Test accuracy for DeepFool under robust classifier: %.3f%%" % (acc))
#print('8/6')
##DeeepFool vs Feature Squeezing
#classifier = KerasClassifier(clip_values=(0, 1), model=model, use_logits=False)
#fs = FeatureSqueezing(bit_depth=4, clip_values=(0, 1))
#img_def, _ = fs(img_adv_df)
#pred_def = model.predict(img_def)
#label_def = np.argmax(pred_def, axis=1)
#confidence_def = pred_def[:, label_def]
#preds = np.argmax(classifier.predict(x_test[:100]), axis=1)
#acc = np.sum(preds == np.argmax(y_test[:100], axis=1)) 
#print("Test accuracy for Feature Squeezing for normal instances: %.3f%%" % (acc))
## Generate PGD attack 
#adv = pgd(classifier, targeted=False, eps_step=.1, eps=.3)
#img_adv = adv.generate(x_test[0:100])
#fs = FeatureSqueezing(bit_depth=4, clip_values=(0, 1))
#img_def, _ = fs(img_adv)
#pred_def = model.predict(img_def)
#label_def = np.argmax(pred_def, axis=1)
#confidence_def = pred_def[:, label_def]
#acc_fs = np.sum(pred_def == np.argmax(y_test[:100], axis=1)) 
#print("Accuracy for feature squeezing with white PGD attack: %.3f%%" % (acc_fs))
#
## Generate CW attack 
#adv = cw(classifier, targeted=False)
#img_adv = adv.generate(x_test[0:100])
#fs = FeatureSqueezing(bit_depth=4, clip_values=(0, 1))
#img_def, _ = fs(img_adv)
#pred_def = model.predict(img_def)
#label_def = np.argmax(pred_def, axis=1)
#confidence_def = pred_def[:, label_def]
#
#acc_fs = np.sum(pred_def == np.argmax(y_test[:100], axis=1)) 
#print("Accuracy for feature squeezing with white CW attack: %.3f%%" % (acc_fs))
#
## Generate FGSM attack 
#adv_fgsm = fgsm(classifier, eps=epsilon)
#img_adv_fgsm = adv_fgsm.generate(x_test[0:100])
#fs = FeatureSqueezing(bit_depth=4, clip_values=(0, 1))
#img_def, _ = fs(img_adv_fgsm)
#pred_def = model.predict(img_def)
#label_def = np.argmax(pred_def, axis=1)
#confidence_def = pred_def[:, label_def]
#
#acc_fs = np.sum(pred_def == np.argmax(y_test[:100], axis=1)) 
#print("Accuracy for feature squeezing with FGSM attack: %.3f%%" % (acc_fs))
# Craft adversarial samples with DeepFool
#adv_crafter = DeepFool(classifier)
#img_adv_df = adv_crafter.generate(x_test[0:100])
#fs = FeatureSqueezing(bit_depth=8, clip_values=(0, 1))
#img_def, _ = fs(img_adv_df)
#pred_def = model.predict(img_def)
#label_def = np.argmax(pred_def, axis=1)
#confidence_def = pred_def[:, label_def]
