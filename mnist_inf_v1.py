from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import os
from os.path import abspath
sys.path.append(abspath('.'))
import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import load_model
from art.classifiers import KerasClassifier
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as pgd
from art.attacks.carlini import CarliniLInfMethod as cw
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod as bim
from parallel_model import make_model_parallel
from scipy.linalg import norm
from sklearn.metrics import confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt
import pickle
import time
import random
from art.attacks.deepfool import DeepFool
random.seed(1)
import keras
'''
Implement proposed approach on MNIST dataset
'''
print('2 cnn, 32, 64 filters respectively; kernel-size: 5; dropout =0.25')
start_time = time.time()
def build_discriminator():
    model =Sequential([
            Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)),
            Conv2D(64, (5,5), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(1024, activation='relu'), #128
            Dropout(0.25),
            Dense(11, activation='softmax')], name = 'Detector')

    image = Input(shape=(28, 28, 1))
    output = model(image)
    make_model_parallel(model)
    return Model(image, output)
steps =1000 
batch_size=60
soft_label = 0.95 # 71% CW for soft_label=0.9; 
print('soft_label ', soft_label)
if __name__ == '__main__':
    latent_dim = 64
    # Build and compile the discriminator
    discriminator = build_discriminator()
#    discriminator.compile(loss='categorical_crossentropy',
#                          optimizer=Adam(lr=0.0002, beta_1=0.5),
#                          metrics=['accuracy'])
    discriminator.compile(loss=keras.losses.categorical_crossentropy, optimizer=Adam(lr = 0.01), 
                          metrics=['accuracy'])
    # Build the generator
    generator = load_model('saved_models/gan_mnist.h5')
# train the GAN system
    #(x_train, y_train), (x_test, y_test), min_, max_ = load_dataset(str('mnist'))
    with open('x_train_mnist.pickle', 'rb') as infile:
        x_train = pickle.load(infile)
    with open('y_train_mnist.pickle', 'rb') as infile:
        y_train = pickle.load(infile)
    min_ =0
    max_ =1
    with open('x_test_mnist.pickle', 'rb') as infile:
        x_test = pickle.load(infile)
    with open('y_test_mnist.pickle', 'rb') as infile:
        y_test = pickle.load(infile)
    y_train=np.insert(y_train, 10, 0, axis=1)# add a column of 0 to indicate real image
    # Define soft_label
    for i in np.arange(y_train.shape[0]):
        y_train[i][np.argmax(y_train, axis =1)[i]]=soft_label
    y_train=y_train+1/11*(1-soft_label)*np.ones((y_train.shape))
    latent_dim = generator.input_shape[1]
    
    for step in range(steps):
        # Train the detector
        # 1. Select a random batch of images
        ind_real = np.random.randint(0, x_train.shape[0], batch_size)
        real_images = x_train[ind_real]
        y_real_label = y_train[ind_real] 
        # 2. Train and evaluate a baseline classifier
        if step ==0:
            classifier_model = load_model('saved_models/mnist_cnn_original.h5')     
#            classifier_model = load_model('saved_models/proj_classifier.h5')     
      # KerasClassifier: Implementation of the scikit-learn classifier API for Keras.
            classifier = KerasClassifier(clip_values=(min_, max_), model=classifier_model, use_logits=False)
        else:
            classifier = KerasClassifier(clip_values=(min_, max_), model=discriminator, use_logits=False)
        # 3. Generate some adversarial samples:
        mean = np.random.poisson(0.01)
        eta=np.random.uniform(0.003,0.3,1)[0]
        
        n_set = 3 # number of the subset of the generator images
        attack_bim = bim(classifier, eps=0.3, eps_step=0.1, max_iter=40)            
        #attack_cw = cw(classifier, targeted=False, batch_size=int(batch_size/n_set))
        a=np.random.choice(y_train.shape[0], batch_size) # return np.array of the selected x_train to be fake
        fake=np.zeros((batch_size, 28, 28, 1))
        fake[:int(batch_size/n_set)] = np.random.normal(mean, eta, (int(batch_size/n_set), 28, 28, 1)) + x_train[a[:int(batch_size/n_set)]]
        fake[int(batch_size/n_set):2*int(batch_size/n_set)] = attack_bim.generate(x_train[a[int(batch_size/n_set):2*int(batch_size/n_set)]])
        # Random batch of noise input for generator
        noise_intro = np.random.normal(mean, 1, (batch_size-(n_set-1)*int(batch_size/n_set), latent_dim))
        # Generate a batch of new images        
        fake[(n_set-1)*int(batch_size/n_set):] = generator.predict(noise_intro)
        # Redefine the labels
        y_adv = y_train[a]
        perturbation = np.zeros(batch_size) # record the size of perturbation (2-norm)
        for i in np.arange((n_set-1)*int(batch_size/n_set)):
            diff = x_train[a[i]]-fake[i]
            diff = diff.reshape((28,28))
            perturbation[i] = norm(diff)/28
            if perturbation[i] >0.01:
                y_adv[i, -1] = y_adv[i, -1]+soft_label/4
                y_adv[i, np.argmax(y_adv[i])]=y_adv[i, np.argmax(y_adv[i])]-soft_label/4
        # option 1:
        y_adv[(n_set-1)*int(batch_size/n_set):,-1] = soft_label
        y_adv[(n_set-1)*int(batch_size/n_set):,:-1]= (1-soft_label)/10 
        # Discriminator Loss Function
        discriminator_loss = discriminator.train_on_batch(real_images, y_real_label)# return loss and accuracy
        discriminator_fake_loss = discriminator.train_on_batch(fake, y_adv)
        discriminator_loss = 0.5*np.add(discriminator_loss, discriminator_fake_loss)
        # Display progress
        if step % 10 == 0:
            print("%d [Discriminator loss: %.4f%%, acc.: %.2f%%]" %
              (step, discriminator_loss[0], 100 * discriminator_loss[1]))
print("--- %s seconds ---" % (time.time() - start_time))
#save the model
model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + 'final_mnist_44.h5'
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_path = os.path.join(save_dir, model_name)
print('Saved trained model at %s ' % model_path)
discriminator.save(model_path)

# Evaluate the classifier on the normal test set 
adv_sample = 100
# add test image noise
mean = np.random.poisson(.01)
x_test_noise = x_test[:adv_sample] + np.random.normal(mean, 0.01,(adv_sample, 28, 28,1)) 
preds_noise = np.argmax(discriminator.predict(x_test_noise), axis=1)
acc_noise = np.sum(preds_noise == np.argmax(y_test[:adv_sample], axis=1)) / adv_sample
print("\nTest accuracy with noise: %.1f%%" % (acc_noise * 100))
preds = np.argmax(discriminator.predict(x_test[:adv_sample]), axis=1)
acc = np.sum(preds == np.argmax(y_test[:adv_sample], axis=1)) / adv_sample
print("\nTest accuracy: %.1f%%" % (acc * 100))
#    np.savetxt(model_name + "confusion_matrix.csv", confusion_matrix(np.argmax(y_test[:adv_sample], axis=1).tolist(), preds.tolist()), delimiter=",")#, fmt='%int')

np.savetxt(model_name + "confusion_matrix.csv", confusion_matrix(preds_noise.tolist(), preds.tolist()), delimiter=",")#, fmt='%int')
# Craft adversarial samples with FGSM (adv_sample:adv_sample*2 )
epsilon = .1  # Maximum perturbation
print('Maximum perturbation (epsilon) for testing is ', epsilon)
classifier = KerasClassifier(clip_values=(min_, max_), model=discriminator, use_logits=False)
adv_crafter = FastGradientMethod(classifier, eps=epsilon)
x_test_adv = adv_crafter.generate(x=x_test[adv_sample:adv_sample*2])
# Evaluate the classifier on the adversarial examples
# add test image noise
x_test_adv_random=x_test_adv+np.random.normal(mean,0.01,x_test_adv.shape) 
preds_random = np.argmax(classifier.predict(x_test_adv_random), axis=1)
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
TP=0
TP_random = 0
TP_comb =0
for i in np.arange(adv_sample):
    diff = x_test[i+adv_sample]-x_test_adv[i]
    diff_random = x_test[i+adv_sample]-x_test_adv_random[i]
    diff = diff.reshape((28,28))
    diff_random = diff_random.reshape((28,28))
    perturbation = norm(diff)/28
    perturbation_random = norm(diff_random)/28
    T=0 # indicator variable
    if ((preds[i] == np.argmax(y_test[i+adv_sample])) |  ((perturbation >0.01) & preds[i]==10)):
        TP=TP+1
        T=1 
    if ((preds_random[i] == np.argmax(y_test[i+adv_sample])) |  ((perturbation_random >0.01) & preds_random[i]==10)):
        TP_random=TP_random +1 
        T=1 #indicator variable
    if (preds_random[i] !=preds[i]):
        TP_comb=TP_comb+1      
    else:
        TP_comb=TP_comb +T
TPR_random = TP_random/ adv_sample       
TPR_comb = TP_comb/adv_sample    
TPR = TP/ adv_sample
print("\nTPR for FGSM: %.1f%%" % (TPR * 100))
print("\nTPR for FGSM when combining: %.1f%%" % (TPR_comb * 100)) 
print("\nTPR for FGSM when random noise is added: %.1f%%" % (TPR_random * 100)) 
# Craft adversarial samples with C&W (adv_sample_cw = 100)
adv_crafter = cw(classifier, targeted=False, batch_size=100)
adv_sample_cw = 100
x_test_adv_cw = adv_crafter.generate(x=x_test[2*adv_sample:2*adv_sample+adv_sample_cw ])
# Evaluate the classifier on the adversarial examples
# add test image noise
x_test_adv_cw_random=x_test_adv_cw+np.random.normal(mean,0.01,x_test_adv_cw.shape) 
preds_cw_random = np.argmax(classifier.predict(x_test_adv_cw_random), axis=1)
preds_cw = np.argmax(classifier.predict(x_test_adv_cw), axis=1)
y_adv = y_test[2*adv_sample:2*adv_sample+adv_sample_cw]
TP_cw_random=0
TP_cw=0
TP_comb_cw =0
for i in np.arange(adv_sample_cw):
    diff_random = x_test[i+2*adv_sample]-x_test_adv_cw_random[i]
    diff_random = diff_random.reshape((28,28))
    perturbation_random = norm(diff_random)/28
    diff = x_test[i+2*adv_sample]-x_test_adv_cw[i]
    diff = diff.reshape((28,28))
    perturbation = norm(diff)/28
    Tcw=0 # indicator variable
    if ((preds_cw_random[i] == np.argmax(y_test[i+2*adv_sample])) |  ((perturbation_random >0.01) & preds_cw_random[i]==10)):
        TP_cw_random=TP_cw_random+1
        Tcw=1 
    if ((preds_cw[i] == np.argmax(y_test[i+2*adv_sample])) |  ((perturbation >0.01) & preds_cw[i]==10)):
        TP_cw=TP_cw+1
        Tcw=1
    if (preds_cw_random[i] !=preds_cw[i]):
        TP_comb_cw=TP_comb_cw+1
    else:
        TP_comb_cw=TP_comb_cw+Tcw
TPR_cw_random = TP_cw_random/ adv_sample_cw
print("\nTPR for C&W when random noise is added: %.1f%%" % (TPR_cw_random * 100)) 
TPR_cw_comb = TP_comb_cw/adv_sample_cw
TPR_cw = TP_cw/ adv_sample_cw
print("\nTPR for C&W: %.1f%%" % (TPR_cw * 100)) 
print("\nTPR for C&W when combining: %.1f%%" % (TPR_cw_comb * 100)) 

# Craft adversarial samples with PGD (adv_sample_cw = 100)
adv_crafter = pgd(classifier, eps=epsilon, eps_step = epsilon/3)
x_test_adv_pgd = adv_crafter.generate(x=x_test[3*adv_sample:3*adv_sample+adv_sample_cw ])
# Evaluate the classifier on the adversarial examples
# add test image noise
x_test_adv_pgd_random=x_test_adv_pgd+np.random.normal(mean,0.01,x_test_adv_pgd.shape) 
preds_pgd_random = np.argmax(classifier.predict(x_test_adv_pgd_random), axis=1)
preds_pgd = np.argmax(classifier.predict(x_test_adv_pgd), axis=1)
y_adv = y_test[3*adv_sample:3*adv_sample+adv_sample_cw]
TP_pgd_random=0
TP_pgd=0
TP_comb_pgd =0
for i in np.arange(adv_sample_cw):
    diff_random = x_test[i+3*adv_sample]-x_test_adv_pgd_random[i]
    diff_random = diff_random.reshape((28,28))
    perturbation_random = norm(diff_random)/28
    diff = x_test[i+3*adv_sample]-x_test_adv_pgd[i]
    diff = diff.reshape((28,28))
    perturbation = norm(diff)/28
    Tpgd=0 # indicator variable
    if ((preds_pgd_random[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation_random >0.01) & preds_pgd_random[i]==10)):
        TP_pgd_random=TP_pgd_random+1
        Tpgd=1 
    if ((preds_pgd[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation >0.01) & preds_pgd[i]==10)):
        TP_pgd=TP_pgd+1
        Tpgd=1
    if (preds_pgd_random[i] !=preds_pgd[i]):
        TP_comb_pgd=TP_comb_pgd+1
    else:
        TP_comb_pgd=TP_comb_pgd+Tpgd
TPR_pgd_random = TP_pgd_random/ adv_sample_cw
print("\nTPR for PGD when random noise is added: %.1f%%" % (TPR_pgd_random * 100)) 
TPR_pgd_comb = TP_comb_pgd/adv_sample_cw
TPR_pgd = TP_pgd/ adv_sample_cw
print("\nTPR for PGD: %.1f%%" % (TPR_pgd * 100)) 
print("\nTPR for PGD when combining: %.1f%%" % (TPR_pgd_comb * 100)) 

# Craft adversarial samples using DeepFool 
attack_DeepFool = DeepFool(classifier)
x_test_adv_df = attack_DeepFool.generate(x=x_test[3*adv_sample:3*adv_sample+adv_sample_cw ])
# Evaluate the classifier on the adversarial examples
# add test image noise
x_test_adv_df_random=x_test_adv_df+np.random.normal(mean,0.01,x_test_adv_df.shape) 
preds_df_random = np.argmax(classifier.predict(x_test_adv_df_random), axis=1)
preds_df = np.argmax(classifier.predict(x_test_adv_df), axis=1)
y_adv = y_test[3*adv_sample:3*adv_sample+adv_sample_cw]
TP_df_random=0
TP_df=0
TP_comb_df =0
for i in np.arange(adv_sample_cw):
    diff_random = x_test[i+3*adv_sample]-x_test_adv_df_random[i]
    diff_random = diff_random.reshape((28,28))
    perturbation_random = norm(diff_random)/28
    diff = x_test[i+3*adv_sample]-x_test_adv_df[i]
    diff = diff.reshape((28,28))
    perturbation = norm(diff)/28
    Tdf=0 # indicator variable
    if ((preds_df_random[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation_random >0.01) & preds_df_random[i]==10)):
        TP_df_random=TP_df_random+1
        Tdf=1 
    if ((preds_df[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation >0.01) & preds_df[i]==10)):
        TP_df=TP_df+1
        Tdf=1
    if (preds_df_random[i] !=preds_df[i]):
        TP_comb_df=TP_comb_df+1
    else:
        TP_comb_df=TP_comb_df+Tdf
TPR_df_random = TP_df_random/ adv_sample_cw
print("\nTPR for DeepFool when random noise is added: %.3f%%" % (TPR_df_random * 100)) 
TPR_df_comb = TP_comb_df/adv_sample_cw
TPR_df = TP_df/ adv_sample_cw
print("\nTPR for DeepFool: %.3f%%" % (TPR_df * 100)) 
print("\nTPR for DeepFool when combining: %.3f%%" % (TPR_df_comb * 100)) 

# Compare the performance of the original and the robust classifier over a range of `eps` values:
eps_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.12, 0.14, 0.16]
nb_correct_original = []
nb_correct_robust = []
#classifier_model = load_model('/home/jinglin/adversarial-robustness-toolbox/saved_models/proj_classifier.h5')  
classifier_model = load_model('saved_models/mnist_cnn_original.h5')     
original_classifier = KerasClassifier(clip_values=(0, 1), model=classifier_model, use_logits=False)
x_test_pred = np.argmax(original_classifier.predict(x_test[:100]), axis=1) 
nb_correct_pred = np.sum(x_test_pred == np.argmax(y_test[:100], axis=1))
# Evaluate the robust classifier's performance on the original test data:
x_test_robust_pred = np.argmax(classifier.predict(x_test[:100]), axis=1)
nb_correct_robust_pred = np.sum(x_test_robust_pred == np.argmax(y_test[:100], axis=1))
adv_plot = FastGradientMethod(original_classifier, eps=0.5)
for eps in eps_range:
    adv_plot.set_params(**{'eps': eps})
    x_test_adv = adv_plot.generate(x_test[:100])#fast gradient method
    x_test_adv_robust = adv_crafter.generate(x_test[:100])
    
    x_test_adv_pred = np.argmax(original_classifier.predict(x_test_adv), axis=1)
    nb_correct_original += [np.sum(x_test_adv_pred == np.argmax(y_test[:100], axis=1))]
    
    x_test_adv_robust_pred = np.argmax(classifier.predict(x_test_adv_robust), axis=1)
    nb_correct_robust += [np.sum(x_test_adv_robust_pred == np.argmax(y_test[:100], axis=1))]

eps_range = [0] + eps_range
nb_correct_original = [nb_correct_pred] + nb_correct_original
nb_correct_robust = [nb_correct_robust_pred] + nb_correct_robust

fig, ax = plt.subplots()
ax.plot(np.array(eps_range), np.array(nb_correct_original), 'b--', label='Original classifier')
ax.plot(np.array(eps_range), np.array(nb_correct_robust), 'r--', label='Robust classifier')
legend = ax.legend(loc='upper center', shadow=True, fontsize='large')
legend.get_frame().set_facecolor('#00FFCC')
plt.xlabel('Attack strength (eps)')
plt.ylabel('Correct predictions')
plt.savefig(model_name + '_foo.png')
print('norm()/28')
print('mnist_inf.py')
#    print('attacker = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.1, max_iter=40)')
print('bim(classifier, eps=0.3, eps_step=0.1, max_iter=40)')
print('discriminator with a CNN with 64 (5, 5) filters and one fully connected layers with 11 neurons: not good')
