import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import os
import time
#from urllib.request import urlopen,urlretrieve
#from PIL import Image
#from tqdm import tqdm_notebook
from numpy.random import permutation
from sklearn.utils import shuffle
import cv2
#from resnets_utils import *
import tensorflow
from tensorflow.keras.models import load_model
from sklearn.datasets import load_files   
from keras.utils import np_utils
from glob import glob
from tensorflow.keras import applications
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential,Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D,GlobalAveragePooling2D
from tensorflow.keras.callbacks import TensorBoard,ReduceLROnPlateau,ModelCheckpoint
from keras_cv_attention_models.volo import *


def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    return resized

def load_train(path):
    '''Give path of the dataset .csv file of training data below'''
    '''Give path of the dataset .csv file of training data below'''
    #df = pd.read_csv(r'/home/marcelo/Downloads/v1_cam1_no_split/Train_data_list.csv')
    df = pd.read_csv(os.path.join(path, 'train.csv'))
    x = df.iloc[:,3]
    y = df.iloc[:,2]
    X_train = []
    Y_train = []
    print('Read train images')
    for i in range (0,len(x)):
        fl=str(x[i]).replace('img', '/home/marcelo/Documentos/Distracted-Driver-Detection/imgs/train/'+ str(y[i] + '/img'))
        #fl = str(x[i])
        img = get_im_cv2(fl)
        X_train.append(img)
        Y_train.append(str(y[i]).replace('c', ''))
    return X_train, Y_train

def load_valid_1(path):
    '''Give path of .csv file of test data below'''
    #df = pd.read_csv(r'/home/marcelo/Downloads/v1_cam1_no_split/Test_data_list.csv')
    df = pd.read_csv(os.path.join(path, 'valid.csv'))
    x = df.iloc[:,3]
    y = df.iloc[:,2]
    X_valid = []
    Y_valid = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=str(x[i]).replace('img', '/home/marcelo/Documentos/Distracted-Driver-Detection/imgs/train/'+ str(y[i] + '/img')) 
        #fl = str(x[i])
        img = get_im_cv2(fl)
        X_valid.append(img)
        Y_valid.append(str(y[i]).replace('c', ''))
    return X_valid, Y_valid

def load_test_1(path):
    '''Give path of .csv file of test data below'''
    df = pd.read_csv(os.path.join(path, 'test.csv'))
    x = df.iloc[:,3]
    y = df.iloc[:,2]
    X_valid = []
    Y_valid = []
    print('Read test images')
    for i in range (0,len(x)):
        fl=str(x[i]).replace('img', '/home/marcelo/Documentos/Distracted-Driver-Detection/imgs/train/'+ str(y[i] + '/img'))
        #fl = str(x[i])
        img = get_im_cv2(fl)
        X_valid.append(img)
        Y_valid.append(str(y[i]).replace('c', ''))
    return X_valid, Y_valid

def get_im_cv2(path):
    img = cv2.imread(path)
    resized = cv2.resize(src=img, dsize=(128, 128), interpolation=cv2.INTER_LINEAR)
    return resized





def read_and_normalize_train_data(path):
    
    train_data, train_target= load_train(path)
    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)
    
    print('Reshape...')
    train_data = train_data.transpose((0, 1, 2, 3))

    # Normalise the train data
    print('Convert to float...')
    train_data = train_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]
    print('Substract 0...')
    train_data[:, :, :, 0] -= mean_pixel[0]
    
    print('Substract 1...')
    train_data[:, :, :, 1] -= mean_pixel[1]

    print('Substract 2...')
    train_data[:, :, :, 2] -= mean_pixel[2]

    train_target = np_utils.to_categorical(train_target, 10)
    
    # Shuffle experiment START !!
    perm = permutation(len(train_target))
    train_data = train_data[perm]
    train_target = train_target[perm]
    # Shuffle experiment END !!
    
    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target

def read_and_normalize_valid_data(path):
    start_time = time.time()
    
    test_data, test_target = load_valid_1(path) #x_test, y_test
    
    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 1, 2, 3))

    # Normalise the test data data

    test_data = test_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]

    test_data[:, :, :, 0] -= mean_pixel[0]

    test_data[:, :, :, 1] -= mean_pixel[1]

    test_data[:, :, :, 2] -= mean_pixel[2]

    test_target = np_utils.to_categorical(test_target, 10)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_target

def read_and_normalize_test_data(path):
    start_time = time.time()

    test_data, test_target = load_test_1(path) #x_test, y_test

    test_data = np.array(test_data, dtype=np.uint8)
    test_data = test_data.transpose((0, 1, 2, 3))

    # Normalise the test data data

    test_data = test_data.astype('float16')
    mean_pixel = [80.857, 81.106, 82.928]

    test_data[:, :, :, 0] -= mean_pixel[0]

    test_data[:, :, :, 1] -= mean_pixel[1]

    test_data[:, :, :, 2] -= mean_pixel[2]

    test_target = np_utils.to_categorical(test_target, 10)
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_target




# Normalize image vectors
#X_train = X_train/255.
#X_valid = X_valid/255.

# Convert training and test labels to one hot matrices
#Y_train = convert_to_one_hot(Y_train_orig, 6).T
#Y_test = convert_to_one_hot(Y_test_orig, 6).T



def train_model_volo(path):

    X_train, Y_train = read_and_normalize_train_data(path)
    X_valid, Y_valid = read_and_normalize_valid_data(path)
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_valid.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_valid shape: " + str(X_valid.shape))
    print ("Y_test shape: " + str(Y_valid.shape))
    #Data augmentation

    datagen = ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                shear_range=0.2
                )
        
    datagen.fit(X_train)

    img_height,img_width = 128,128 
    num_classes = 10
    #If imagenet weights are being loaded, 
    #input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    base_model = VOLO_d2(input_shape= (128, 128, 3), num_classes= 10)#applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    batch_size = 64
    nb_epoch =256
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    from tensorflow.keras.optimizers import SGD, Adam
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.fit(X_train, Y_train, epochs = 100, batch_size = 64)
    if not os.path.isdir(os.path.join(os.getcwd(), 'Checkpoint')):
            os.mkdir(os.path.join(os.getcwd(), 'Checkpoint'))
        
    weights_path=os.path.join(os.getcwd(), 'Checkpoint', 'weights.h5')       
    callbacks = [ModelCheckpoint(weights_path, monitor='val_accuracy', save_weights_only=True, verbose=1, save_best_only = True)]

    #with tf.device('/GPU:0'):
    hist=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)


    pd.DataFrame(hist.history).to_csv(os.path.join(os.getcwd(), 'Checkpoint', 'try_hist.csv'))

    predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
    cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
    
    ppath=os.path.join(os.path.join(os.getcwd(),'Checkpoint','confusion_mat.npy'))
    np.save(ppath, cm1)

def test_model_volo(path_weight, path_to_test_dataset):
    X_test, Y_test = read_and_normalize_test_data(path_to_test_dataset)
    base_model = VOLO_d2(input_shape= (128, 128, 3), num_classes= 10)#applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    num_classes = 10
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    batch_size = 64
    nb_epoch =256
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    from tensorflow.keras.optimizers import SGD, Adam
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(path_weight)
    predictions_valid = model.predict(X_test.astype('float32'), batch_size=64, verbose=1)
    cm1=confusion_matrix(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1))
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    x = classification_report(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), target_names = labels)
    y = f1_score(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), average= 'macro')
    print("None {}".format(x))
    print("EI ", y)
    ppath=os.path.join(os.path.join(os.getcwd(),'confusion_mat_test.npy'))
    np.save(ppath, cm1)
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    print('Restored model, loss: {:5.2f}%'.format(loss))

def train_model_efficientNet(path):

    X_train, Y_train = read_and_normalize_train_data(path)
    X_valid, Y_valid = read_and_normalize_valid_data(path)
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_valid.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_valid shape: " + str(X_valid.shape))
    print ("Y_test shape: " + str(Y_valid.shape))
    #Data augmentation

    datagen = ImageDataGenerator(
                width_shift_range=0.2,
                height_shift_range=0.2,
                zoom_range=0.2,
                shear_range=0.2
                )
        
    datagen.fit(X_train)

    img_height,img_width = 128,128 
    num_classes = 10
    #If imagenet weights are being loaded, 
    #input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
    #base_model = EfficientNetV2B2(input_shape= (128, 128, 3), num_classes= 10)#applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))
    base_model = EfficientNetV2M(input_shape= (128, 128, 3), num_classes= 10)#applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    batch_size = 64
    nb_epoch =256
    from sklearn.metrics import confusion_matrix
    from tensorflow.keras.optimizers import SGD, Adam
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    #model.fit(X_train, Y_train, epochs = 100, batch_size = 64)
    if not os.path.isdir(os.path.join(os.getcwd(), 'Checkpoint')):
            os.mkdir(os.path.join(os.getcwd(), 'Checkpoint'))
        
    weights_path=os.path.join(os.getcwd(), 'Checkpoint', 'weights.h5')       
    callbacks = [ModelCheckpoint(weights_path, monitor='val_accuracy', save_weights_only=True, verbose=1, save_best_only = True)]
    model.summary()
    #with tf.device('/GPU:0'):
    hist=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
            verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)


    pd.DataFrame(hist.history).to_csv(os.path.join(os.getcwd(), 'Checkpoint', 'try_hist.csv'))

    predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
    cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
    ppath=os.path.join(os.path.join(os.getcwd(),'Checkpoint','confusion_mat.npy'))
    np.save(ppath, cm1)

def test_model_efficientNet(path_weight, path_to_test_dataset):
    X_test, Y_test = read_and_normalize_test_data(path_to_test_dataset)
    base_model = EfficientNetV2M(input_shape= (128, 128, 3), num_classes= 10)#applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    num_classes = 10
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    batch_size = 64
    nb_epoch =256
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    from tensorflow.keras.optimizers import SGD, Adam
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(path_weight)
    predictions_valid = model.predict(X_test.astype('float32'), batch_size=64, verbose=1)
    cm1=confusion_matrix(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1))
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    x = classification_report(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), target_names = labels)
    y = f1_score(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), average= 'macro')
    print("None {}".format(x))
    print("EI ", y)
    ppath=os.path.join(os.path.join(os.getcwd(),'confusion_mat_test.npy'))
    np.save(ppath, cm1)
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    print('Restored model, loss: {:5.2f}%'.format(loss))


def train_model_efficientNet(path):
	X_train, Y_train = read_and_normalize_train_data("/home/marcelo/Documentos/Distracted-Driver-Detection/imgs")
	X_valid, Y_valid = read_and_normalize_valid_data("/home/marcelo/Documentos/Distracted-Driver-Detection/imgs")
	# Normalize image vectors
	#X_train = X_train/255.
	#X_valid = X_valid/255.

	# Convert training and test labels to one hot matrices
	#Y_train = convert_to_one_hot(Y_train_orig, 6).T
	#Y_test = convert_to_one_hot(Y_test_orig, 6).T

	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_valid.shape[0]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_valid shape: " + str(X_valid.shape))
	print ("Y_test shape: " + str(Y_valid.shape))
	#Data augmentation
		
	datagen = ImageDataGenerator(
		          width_shift_range=0.2,
		          height_shift_range=0.2,
		          zoom_range=0.2,
		          shear_range=0.2
		          )
		
	datagen.fit(X_train)

	img_height,img_width = 128,128 
	num_classes = 10
	#If imagenet weights are being loaded, 
	#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
	base_model = applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.7)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	model = Model(inputs = base_model.input, outputs = predictions)
	batch_size = 64
	nb_epoch =256
	from sklearn.metrics import confusion_matrix
	from tensorflow.keras.optimizers import SGD, Adam
	# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.0001)
	model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
	#model.fit(X_train, Y_train, epochs = 100, batch_size = 64)
	if not os.path.isdir(os.path.join(os.getcwd(), 'Checkpoint')):
		    os.mkdir(os.path.join(os.getcwd(), 'Checkpoint'))
		
	weights_path=os.path.join(os.getcwd(), 'Checkpoint', 'weights.h5')       
	callbacks = [ModelCheckpoint(weights_path, monitor='val_accuracy', save_weights_only=True, verbose=1, save_best_only = True)]

	#with tf.device('/GPU:0'):
	hist=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
		            steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
		    verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)


	pd.DataFrame(hist.history).to_csv(os.path.join(os.getcwd(), 'Checkpoint', 'try_hist.csv'))

	predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
	cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
	ppath=os.path.join(os.path.join(os.getcwd(),'Checkpoint','confusion_mat.npy'))
	np.save(ppath, cm1)

def train_model_resnet(path):
	X_train, Y_train = read_and_normalize_train_data(path)
	X_valid, Y_valid = read_and_normalize_valid_data(path)
	# Normalize image vectors
	#X_train = X_train/255.
	#X_valid = X_valid/255.

	# Convert training and test labels to one hot matrices
	#Y_train = convert_to_one_hot(Y_train_orig, 6).T
	#Y_test = convert_to_one_hot(Y_test_orig, 6).T

	print ("number of training examples = " + str(X_train.shape[0]))
	print ("number of test examples = " + str(X_valid.shape[0]))
	print ("X_train shape: " + str(X_train.shape))
	print ("Y_train shape: " + str(Y_train.shape))
	print ("X_valid shape: " + str(X_valid.shape))
	print ("Y_test shape: " + str(Y_valid.shape))
	#Data augmentation
		
	datagen = ImageDataGenerator(
		          width_shift_range=0.2,
		          height_shift_range=0.2,
		          zoom_range=0.2,
		          shear_range=0.2
		          )
		
	datagen.fit(X_train)

	img_height,img_width = 128,128 
	num_classes = 10
	#If imagenet weights are being loaded, 
	#input must have a static square shape (one of (128, 128), (160, 160), (192, 192), or (224, 224))
	base_model = applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

	x = base_model.output
	x = GlobalAveragePooling2D()(x)
	x = Dropout(0.7)(x)
	predictions = Dense(num_classes, activation= 'softmax')(x)
	model = Model(inputs = base_model.input, outputs = predictions)
	batch_size = 64
	nb_epoch =256
	from sklearn.metrics import confusion_matrix
	from tensorflow.keras.optimizers import SGD, Adam
	# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
	adam = Adam(lr=0.0001)
	model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
	#model.fit(X_train, Y_train, epochs = 100, batch_size = 64)
	if not os.path.isdir(os.path.join(os.getcwd(), 'Checkpoint')):
		    os.mkdir(os.path.join(os.getcwd(), 'Checkpoint'))
		
	weights_path=os.path.join(os.getcwd(), 'Checkpoint', 'weights.h5')       
	callbacks = [ModelCheckpoint(weights_path, monitor='val_accuracy', save_weights_only=True, verbose=1, save_best_only = True)]

	#with tf.device('/GPU:0'):
	hist=model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
		            steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch,
		    verbose=1, validation_data=(X_valid, Y_valid), callbacks=callbacks)


	pd.DataFrame(hist.history).to_csv(os.path.join(os.getcwd(), 'Checkpoint', 'try_hist.csv'))

	predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=1)
	cm1=confusion_matrix(Y_valid.argmax(axis=1), predictions_valid.argmax(axis=1))
	ppath=os.path.join(os.path.join(os.getcwd(),'Checkpoint','confusion_mat.npy'))
	np.save(ppath, cm1)

def test_model_resnet(path_weight, path_to_test_dataset):
   	X_test, Y_test = read_and_normalize_test_data(path_to_test_dataset)
	base_model = applications.resnet.ResNet152(weights= None, include_top=False, input_shape= (img_height,img_width,3))

    x = base_model.output
    #x = GlobalAveragePooling2D()(x)
    #x = Dropout(0.7)(x)
    num_classes = 10
    predictions = Dense(num_classes, activation= 'softmax')(x)
    model = Model(inputs = base_model.input, outputs = predictions)
    batch_size = 64
    nb_epoch =256
    from sklearn.metrics import confusion_matrix, classification_report, f1_score
    from tensorflow.keras.optimizers import SGD, Adam
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    adam = Adam(lr=0.0001)
    model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(path_weight)
    predictions_valid = model.predict(X_test.astype('float32'), batch_size=64, verbose=1)
    cm1=confusion_matrix(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1))
    labels = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    x = classification_report(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), target_names = labels)
    y = f1_score(Y_test.argmax(axis=1), predictions_valid.argmax(axis=1), average= 'macro')
    print("None {}".format(x))
    print("EI ", y)
    ppath=os.path.join(os.path.join(os.getcwd(),'confusion_mat_test.npy'))
    np.save(ppath, cm1)
    loss, acc = model.evaluate(X_test, Y_test, verbose=2)
    print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
    print('Restored model, loss: {:5.2f}%'.format(loss))

train_model_volo("/home/marcelo/Documentos/Distracted-Driver-Detection/imgs")
test_model_volo("/home/marcelo/Documentos/Distracted-Driver-Detection/pesos_volo_kaggle/weights.h5", "/home/marcelo/Documentos/Distracted-Driver-Detection/imgs")
