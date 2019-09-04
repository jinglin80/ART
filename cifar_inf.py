'''
Implement Proposed approach on Cifar10 Dataset

In order to run the code, please install Adversarial Robustness Toolbox v0.9.0 first (June 2019).

If you are able to load the dataset directly, you do not need to use pickle and can mute the corresponding code and directly use
load_dataset function. 

'''
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sys
import numpy as np
from keras import layers
import os
from keras.preprocessing.image import ImageDataGenerator
import pickle
from os.path import abspath
sys.path.append(abspath('.'))
from keras.models import load_model, Sequential
from art.classifiers import KerasClassifier
from art.attacks.projected_gradient_descent import ProjectedGradientDescent as pgd
from art.attacks.carlini import CarliniLInfMethod as cw
from art.attacks.deepfool import DeepFool
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.iterative_method import BasicIterativeMethod as bim
from parallel_model import make_model_parallel
from scipy.linalg import norm
from sklearn.metrics import confusion_matrix
from art.data_generators import KerasDataGenerator
from os.path import abspath
from datetime import datetime
from keras.optimizers import Adam
import math
import random
random.seed(1)
print('cifar10')
# Constants
train_epochs = 800
for soft_label in np.arange(0.5, 0.96, 0.05):
    print('soft_label ', soft_label)
    perturbation_limit = 0.01 # set the perturbation_limit for assigning the soft label to adversarial example
    print('perturbation limit ', perturbation_limit)
    #print('number_epochs = 83')
    test_noise_added = 0.05 # random magnitude of the test noise added
    print('test_noise_added ', test_noise_added)
              
    if __name__ == '__main__':
        # Build and complie the model
        batch_size = 128
        classifier_model = load_model('saved_models/cifar_resnet.h5')
        print('saved_models/cifar_resnet.h5, original acc=92%')
    #    classifier_model = load_model('saved_models/cifar10_original.h5')
    #    print('cifar10_original.h5 orignal acc =88.11')
    #    print('saved_models/cifar10_original.h5 which is constructed vgg19 34%')
        classifier_model.layers.pop()#-1 drop last dense layer
        discriminator = Sequential()
        discriminator.add(classifier_model)
        discriminator.add(layers.Dense(11,activation='softmax'))
        make_model_parallel(discriminator)
        #KerasClassifier(clip_values=(0,1), model =classifier_model, use_logits = False)
        discriminator.compile(loss='categorical_crossentropy',
                              optimizer=Adam(lr=0.0002, beta_1=0.5),
                              metrics=['accuracy'])
        # The dataset:
        with open('adversarial-robustness-toolbox/dataset/x_train.pickle', 'rb') as infile:
            x_train = pickle.load(infile)
        with open('adversarial-robustness-toolbox/dataset/y_train.pickle', 'rb') as infile:
            y_train = pickle.load(infile)
        y_train=np.insert(y_train, 10, 0, axis=1) # add a column of 0 to indicate real image
        # Define soft_label
        for i in np.arange(y_train.shape[0]):
            y_train[i][np.argmax(y_train, axis =1)[i]]=soft_label
        y_train=y_train+1/11*(1-soft_label)*np.ones((y_train.shape))
        #steps_for_epoch = math.ceil(x_train.shape[0] / batch_size)
        latent_size = 110 # z for generator
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            zca_epsilon=1e-06,  # epsilon for ZCA whitening
            rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
            # randomly shift images horizontally (fraction of total width)
            width_shift_range=0.2,
            # randomly shift images vertically (fraction of total height)
            height_shift_range=0.2,
            shear_range=0.2,  # set range for random shear
            zoom_range=0.2,  # set range for random zoom
            channel_shift_range=0.,  # set range for random channel shifts
            # set mode for filling points outside the input boundaries
            fill_mode='nearest',
            cval=0.,  # value used for fill_mode = "constant"
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,  # randomly flip images
            # set rescaling factor (applied before any other transformation)
            rescale=None,
            # set function that will be applied on each input
            preprocessing_function=None,
            # image data format, either "channels_first" or "channels_last"
            data_format=None,
            # fraction of images reserved for validation (strictly between 0 and 1)
            validation_split=0.0)    
        datagen.fit(x_train)
    
        for step in range(train_epochs):
            if step == 0:
                classifier_model = load_model('saved_models/cifar10_original.h5')
                classifier = KerasClassifier(clip_values=(0,1), model =classifier_model, use_logits = False)
                #classifier = KerasClassifier(clip_values=(0, 1), model=classifier_model, use_logits=False, preprocessing=(0.5, 1))
            else:
                classifier =KerasClassifier(clip_values= (0, 1), model = discriminator, use_logits = False)
    #            classifier = KerasClassifier(clip_values=(0, 1), model=classifier_model, use_logits=False, preprocessing=(0.5, 1))
            # 3. Generate some adversarial samples:
            eta=np.random.uniform(0.01,0.5,1)[0]
            attack_bim = bim(classifier, eps=0.3, eps_step=0.01, max_iter=40)
            n_set = 3 # number of the subset of the generator images
            a=np.random.choice(y_train.shape[0], batch_size) # return np.array of the selected x_train to be fake
            fake=np.zeros((batch_size, 32, 32, 3))
            mean = np.random.poisson(.01)
            fake[:int(batch_size/n_set)] = np.random.normal(mean, eta, (int(batch_size/n_set), 32, 32, 3)) + x_train[a[:int(batch_size/n_set)]]
            fake[int(batch_size/n_set):2*int(batch_size/n_set)] = attack_bim.generate(x_train[a[int(batch_size/n_set):2*int(batch_size/n_set)]])
            #generator images
            ind = batch_size-(n_set-1)*int(batch_size/n_set)
            noise = np.random.normal(0, 0.5, (ind, latent_size))
            generator = load_model('saved_models/gan_cifar10.h5')
            random_label = np.random.randint(10, size=ind).reshape(-1,1)
            fake[(n_set-1)*int(batch_size/n_set):] = generator.predict([noise, random_label]).transpose(0, 2, 3, 1)
            # Redefine the labels
            y_adv = y_train[a]
            perturbation = np.zeros(batch_size) # record the size of perturbation (2-norm)
            for i in np.arange((n_set-1)*int(batch_size/n_set)):
                diff = x_train[a[i]]-fake[i]
                diff = diff.reshape((32,32,3))
                perturbation[i] = norm(diff)/32/math.sqrt(3)
                if perturbation[i] > perturbation_limit:
                    y_adv[i, -1] = y_adv[i, -1] + soft_label/4
                    y_adv[i, np.argmax(y_adv[i])] = y_adv[i, np.argmax(y_adv[i])]-soft_label/4
            # option 1 for GAN images:
            y_adv[(n_set-1)*int(batch_size/n_set):,-1]+=0.7
            y_adv[(n_set-1)*int(batch_size/n_set):,:-1]=0.03
            # Fit
            art_datagen = KerasDataGenerator(datagen.flow(x=x_train, y=y_train, batch_size=batch_size, shuffle=True), size=x_train.shape[0], batch_size=batch_size)
            x_batch, y_batch = art_datagen.get_batch()
    #        discriminator.fit_generator(art_datagen, number_epochs, verbose=0)#0: silent
            discriminator.train_on_batch(fake, y_adv)
            discriminator.train_on_batch(x_batch, y_batch)
            # check progress
            if step % 10 == 0:
                preds = np.argmax(classifier.predict(x_train[:1000]), axis=1)
                acc = np.sum(preds == np.argmax(y_train[:1000], axis=1)) / 1000
                print('step: ', step )
                print("Test accuracy for normal instances: %.3f%%" % (acc * 100))
    # Save model and weights
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = datetime.now().strftime("%Y%m%d-%H%M%S") + ' cifar_final_model.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    discriminator.build(None)
    discriminator.save(model_path)
    print('cifar.py')
    print('Saved trained model at %s ' % model_path)
    
    # Evaluations
    with open('adversarial-robustness-toolbox/dataset/x_test.pickle', 'rb') as infile:
        x_test = pickle.load(infile)
    with open('adversarial-robustness-toolbox/dataset/y_test.pickle', 'rb') as infile:
        y_test = pickle.load(infile)
    
    print('x_train ', x_train.shape)
    print(x_train[0])
    print('x_test ' , x_test.shape)
    print(x_test[:10])
    print('y_test ', y_test.shape)
    print(y_test[:10])
    print('y_train ' , y_train.shape)
    print(y_train[0])
    # 1. Classification accuracy
    adv_sample = 1000
    # add test image noise
    mean = np.random.poisson(.01)
    x_test_noise = x_test[:adv_sample] + np.random.normal(mean, test_noise_added,(adv_sample, 32, 32, 3)) 
    preds = np.argmax(discriminator.predict(x_test[:adv_sample]), axis=1)
    preds_noise = np.argmax(discriminator.predict(x_test_noise), axis=1)
    acc = np.sum(preds == np.argmax(y_test[:adv_sample], axis=1)) / adv_sample
    acc_noise = np.sum(preds_noise == np.argmax(y_test[:adv_sample], axis=1)) / adv_sample
    print("\nTest accuracy with noise added: %.1f%%" % (acc_noise * 100))
    print("\n Test accuracy without attacks: %.1f%%" % (acc * 100))
#    np.savetxt(model_name+ "confusion_matrix.csv", confusion_matrix(np.argmax(y_test[:adv_sample], axis=1).tolist(), preds.tolist()), delimiter=",", fmt='%1.3f')
    np.savetxt(model_name + "confusion_matrix.csv", confusion_matrix(preds_noise.tolist(), preds.tolist()), delimiter=",")#, fmt='%int')

    # 2. Craft adversarial samples with FGSM (adv_sample:adv_sample*2 )
    epsilon = .3  # Maximum perturbation (tunable)
    classifier = KerasClassifier(clip_values=(0, 1), model=discriminator, use_logits=False)
    adv_crafter = FastGradientMethod(classifier, eps=epsilon)
    x_test_adv = adv_crafter.generate(x=x_test[adv_sample:adv_sample*2])
    # Evaluate the classifier on the adversarial examples
    # add test image noise
    x_test_adv_random=x_test_adv+np.random.normal(mean, test_noise_added,x_test_adv.shape) 
    preds_random = np.argmax(classifier.predict(x_test_adv_random), axis=1)
    preds = np.argmax(classifier.predict(x_test_adv), axis=1)
    # initialization
    TP=0
    TP_random = 0
    TP_comb =0
    for i in np.arange(adv_sample):
        diff = x_test[i+adv_sample]-x_test_adv[i]
        diff_random = x_test[i+adv_sample]-x_test_adv_random[i]
        diff = diff.reshape((32,32,3))
        diff_random = diff_random.reshape((32,32,3))
        perturbation = norm(diff)/32/math.sqrt(3)
        perturbation_random = norm(diff_random)/32/math.sqrt(3)
        T=0 # indicator variable
        if ((preds[i] == np.argmax(y_test[i+adv_sample])) |  ((perturbation > perturbation_limit) & preds[i]==10)):
            TP=TP+1
            T=1 
        if ((preds_random[i] == np.argmax(y_test[i+adv_sample])) |  ((perturbation_random > perturbation_limit) & preds_random[i]==10)):
            TP_random=TP_random +1 
            T=1 #indicator variable
        if (preds_random[i] !=preds[i]): #for conservative combine
            TP_comb=TP_comb+1      
        else:
            TP_comb=TP_comb +T
    TPR_random = TP_random/ adv_sample       
    TPR_comb = TP_comb/adv_sample    
    TPR = TP/ adv_sample
    print("\nTPR for FGSM: %.3f%%" % (TPR * 100))
    print("\nTPR for FGSM when combining: %.3f%%" % (TPR_comb * 100)) 
    print("\nTPR for FGSM when random noise is added: %.3f%%" % (TPR_random * 100)) 
    # 3. Craft adversarial samples with C&W (adv_sample_cw = 100)
    adv_crafter = cw(classifier, targeted=False, batch_size=100)
    adv_sample_cw = 100
    x_test_adv_cw = adv_crafter.generate(x=x_test[2*adv_sample:2*adv_sample+adv_sample_cw ])
    x_test_adv_cw_random=x_test_adv_cw+np.random.normal(mean, test_noise_added, x_test_adv_cw.shape) 
    preds_cw_random = np.argmax(classifier.predict(x_test_adv_cw_random), axis=1)
    preds_cw = np.argmax(classifier.predict(x_test_adv_cw), axis=1)
    y_adv = y_test[2*adv_sample:2*adv_sample+adv_sample_cw]
    TP_cw_random=0
    TP_cw=0
    TP_comb_cw =0
    for i in np.arange(adv_sample_cw):
        diff_random = x_test[i+2*adv_sample]-x_test_adv_cw_random[i]
        diff_random = diff_random.reshape((32, 32, 3))
        perturbation_random = norm(diff_random)/32/math.sqrt(3)
        diff = x_test[i+2*adv_sample]-x_test_adv_cw[i]
        diff = diff.reshape((32,32,3))
        perturbation = norm(diff)/32/math.sqrt(3)
        Tcw=0 # indicator variable
        if ((preds_cw_random[i] == np.argmax(y_test[i+2*adv_sample])) |  ((perturbation_random > perturbation_limit) & preds_cw_random[i]==10)):
            TP_cw_random=TP_cw_random+1
            Tcw=1 
        if ((preds_cw[i] == np.argmax(y_test[i+2*adv_sample])) |  ((perturbation > perturbation_limit) & preds_cw[i]==10)):
            TP_cw=TP_cw+1
            Tcw=1
        if (preds_cw_random[i] !=preds_cw[i]):
            TP_comb_cw=TP_comb_cw+1
        else:
            TP_comb_cw=TP_comb_cw+Tcw
    TPR_cw_comb = TP_comb_cw/adv_sample_cw
    TPR_cw = TP_cw/ adv_sample_cw
    TPR_cw_random = TP_cw_random/ adv_sample_cw
    print("\nTPR for C&W when random noise is added: %.3f%%" % (TPR_cw_random * 100)) 
    print("\nTPR for C&W: %.3f%%" % (TPR_cw * 100)) 
    print("\nTPR for C&W when combining: %.3f%%" % (TPR_cw_comb * 100)) 
    # Craft adversarial samples with PGD (adv_sample_cw = 100)
    adv_crafter = pgd(classifier, eps=epsilon, eps_step = epsilon/3)
    x_test_adv_pgd = adv_crafter.generate(x=x_test[3*adv_sample:3*adv_sample+adv_sample_cw ])
    # Evaluate the classifier on the adversarial examples
    # add test image noise
    mean = np.random.poisson(.01)
    x_test_adv_pgd_random=x_test_adv_pgd+np.random.normal(mean,0.01,x_test_adv_pgd.shape) 
    preds_pgd_random = np.argmax(classifier.predict(x_test_adv_pgd_random), axis=1)
    preds_pgd = np.argmax(classifier.predict(x_test_adv_pgd), axis=1)
    y_adv = y_test[3*adv_sample:3*adv_sample+adv_sample_cw]
    TP_pgd_random=0
    TP_pgd=0
    TP_comb_pgd =0
    for i in np.arange(adv_sample_cw):
        diff_random = x_test[i+3*adv_sample]-x_test_adv_pgd_random[i]
        diff_random = diff_random.reshape((32, 32, 3))
        perturbation_random = norm(diff_random)/32/math.sqrt(3)
        diff = x_test[i+3*adv_sample]-x_test_adv_pgd[i]
        diff = diff.reshape((32, 32, 3))
        perturbation = norm(diff)/32/math.sqrt(3)
        Tpgd=0 # indicator variable
        if ((preds_pgd_random[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation_random >perturbation_limit) & preds_pgd_random[i]==10)):
            TP_pgd_random=TP_pgd_random+1
            Tpgd=1 
        if ((preds_pgd[i] == np.argmax(y_test[i+3*adv_sample])) |  ((perturbation >perturbation_limit) & preds_pgd[i]==10)):
            TP_pgd=TP_pgd+1
            Tpgd=1
        if (preds_pgd_random[i] !=preds_pgd[i]):
            TP_comb_pgd=TP_comb_pgd+1
        else:
            TP_comb_pgd=TP_comb_pgd+Tpgd
    TPR_pgd_random = TP_pgd_random/ adv_sample_cw
    print("\nTPR for PGD when random noise is added: %.3f%%" % (TPR_pgd_random * 100)) 
    TPR_pgd_comb = TP_comb_pgd/adv_sample_cw
    TPR_pgd = TP_pgd/ adv_sample_cw
    print("\nTPR for PGD: %.3f%%" % (TPR_pgd * 100)) 
    print("\nTPR for PGD when combining: %.3f%%" % (TPR_pgd_comb * 100)) 
# =============================================================================
#     # Craft adversarial samples using DeepFool 
#   check
# =============================================================================
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
    print('cifar_inf.py')
    print('preprocessing=(0.5, 1) deleted')
                    