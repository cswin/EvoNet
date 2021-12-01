"""
Generic setup of the data sources and the model training.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py
and also on
    https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py

"""

#import keras
from keras.datasets       import mnist, cifar10
from keras.models         import Sequential
from keras.layers         import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.callbacks      import EarlyStopping, Callback
from keras.layers         import Conv2D, MaxPooling2D,Activation
from keras                import backend as K
from keras.layers.normalization import BatchNormalization
from random import randint
import os
import logging

import argparse
from glob import glob
from utils import *
import numpy as np
import h5py


parser = argparse.ArgumentParser(description='')
parser.add_argument('--epoch', dest='epoch', type=int, default=10, help='# of epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=128, help='# images in batch')
parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--sigma', dest='sigma', type=int, default=22, help='noise level')

parser.add_argument('--mA', dest='mA', default='20mA', help='CT noise level')

parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
parser.add_argument('--test_dir', dest='test_dir', default='./test', help='test sample are saved here')
parser.add_argument('--eval_set_ct10clean', dest='eval_set_ct10clean', default='ct10_clean', help='dataset for eval in training')
parser.add_argument('--eval_set_ct10noisy', dest='eval_set_ct10noisy', default='ct10_noisy', help='dataset for eval in training')

args = parser.parse_args()


# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=3, verbose=0, mode='auto' )

#patience=5)
#monitor='val_loss',patience=2,verbose=0
#In your case, you can see that your training loss is not dropping - which means you are learning nothing after each epoch.
#It look like there's nothing to learn in this model, aside from some trivial linear-like fit or cutoff value.

def get_cifar10_mlp():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent
    batch_size  = 64
    epochs      = 4
    input_shape = (3072,) #because it's RGB

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test  = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_cifar10_cnn():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10 #dataset dependent
    batch_size = 128
    epochs     = 4

    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    #x._train shape: (50000, 32, 32, 3)
    #input shape (32, 32, 3)
    input_shape = x_train.shape[1:]

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')
    #print('input shape', input_shape)

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_mlp():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes  = 10 #dataset dependent
    batch_size  = 64
    epochs      = 4
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test  = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_cnn():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10 #dataset dependent
    batch_size = 128
    epochs     = 4

    # Input image dimensions
    img_rows, img_cols = 28, 28

    # Get the data.
    # the data, shuffled and split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    #x_train = x_train.reshape(60000, 784)
    #x_test  = x_test.reshape(10000, 784)

    x_train = x_train.astype('float32')
    x_test  = x_test.astype('float32')
    x_train /= 255
    x_test  /= 255

    #print('x_train shape:', x_train.shape)
    #print(x_train.shape[0], 'train samples')
    #print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test  = to_categorical(y_test,  nb_classes)

    # convert class vectors to binary class matrices
    #y_train = keras.utils.to_categorical(y_train, nb_classes)
    #y_test = keras.utils.to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs)

def get_mnist_denoisingcnn():
 
    eval_files = glob('./data/test/mnist/*.jpg'.format(args.eval_set))
    eval_data = load_images(eval_files)  # list of array of different size, 4-D, pixel value range is 0-255

    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
        
        
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
    x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
    
    noise_factor = 25/255;
    x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
    x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
    
    x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return (x_train,x_train_noisy,eval_data,args.batch_size,args.epoch,input_shape)


def get_ct_denoisingcnn():
    """Retrieve the ct dataset and process the data."""
    if K.image_data_format() == 'channels_first':
        input_shape = (1, None, None)
    else:
        input_shape = (None, None, 1)
 
    #load  clean images as label
    
    
   # f = h5py.File('./data/imdb-residual-ct35-training-only.mat', 'r')
    f = h5py.File('./data/imdb-250images.mat', 'r')
#    f = h5py.File('/data/TEST/DnCNN_Training/data/EvoNET11003_MICCAI_N22/imdb-100images.mat', 'r')
    y_train_clean=np.transpose(f['labels'],(0,2,3,1))
    x_train_noisy=np.transpose(f['inputs'],(0,2,3,1))
    
    
#    y_train_clean=np.load('./data/img_clean_ct_clean.npy')
#    y_train_clean = y_train_clean.astype('float32') / 255.
    
#    x_train_noisy=np.load('./data/img_clean_ct_noisy_20mA.npy')
#    x_train_noisy = x_train_noisy.astype('float32') / 255.
    

   # test data during training
    eval_files_clean = glob('./data/test/ct10_clean/*.bmp'.format(args.eval_set_ct10clean))
    eval_data_clean = load_images(eval_files_clean) 
    eval_files_noisy = glob('./data/test/ct10_noisy_'+args.mA+'/*.bmp'.format(args.eval_set_ct10noisy))
    eval_data_noisy = load_images(eval_files_noisy) 
    
    
#    eval_files_clean = glob('./data/test/RLD_10_clean/*.bmp'.format(args.eval_set_ct10clean))
#    eval_data_clean = load_images(eval_files_clean) 
#    eval_files_noisy = glob('./data/test/RLD_10_noisy/*.bmp'.format(args.eval_set_ct10noisy))
#    eval_data_noisy = load_images(eval_files_noisy) 
    
    return (x_train_noisy, y_train_clean, eval_data_clean,eval_data_noisy,args.batch_size,args.epoch,input_shape)



def compile_model_mlp(geneparam, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = geneparam['nb_layers' ]
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer  = geneparam['optimizer' ]

    logging.info("Architecture-----:%d, %s, %s, %d" % (nb_neurons, activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            model.add(Dense(nb_neurons, activation=activation, input_shape=input_shape))
        else:
            model.add(Dense(nb_neurons, activation=activation))

        model.add(Dropout(0.2))  # hard-coded dropout for each layer

    # Output layer.
    model.add(Dense(nb_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])

    return model

def compile_model_cnn(geneparam, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = geneparam['nb_layers' ]
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer  = geneparam['optimizer' ]

    logging.info("Architecture:%d,%s,%s,%d" % (nb_neurons, activation, optimizer, nb_layers))

    model = Sequential()

    # Add each layer.
    for i in range(0,nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons, kernel_size = (3, 3), activation = activation, padding='same', input_shape = input_shape))
        else:
            model.add(Conv2D(nb_neurons, kernel_size = (3, 3), activation = activation))

        if i < 2: #otherwise we hit zero
            model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Dropout(0.2))

    model.add(Flatten())   #
    # now: model.output_shape == (None, 64, 32, 32)
    # now: model.output_shape == (None, 65536)
    model.add(Dense(nb_neurons, activation = activation))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation = 'softmax'))

    #BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    #need to read this paper

    model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model

def compile_model_cnn_denoising(geneparam, input_shape):
    """Compile a sequential model.

    Args:
        genome (dict): the parameters of the genome

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers  = geneparam['nb_layers' ]
    nb_neurons = geneparam['nb_neurons']
    activation = geneparam['activation']
    optimizer  = geneparam['optimizer' ]

    logging.info("***Architecture: nb_layers:%d, nb_neurons:%d, activation:%s, optimizer:%s" % (nb_layers,nb_neurons, activation, optimizer))
    print("***Architecture: nb_layers:%d, nb_neurons:%d, activation:%s, optimizer:%s" % (nb_layers,nb_neurons, activation, optimizer))

    model = Sequential()

    # Add each layer.
    for i in range(0,nb_layers):
        # Need input shape for first layer.
        if i == 0:
            model.add(Conv2D(nb_neurons, kernel_size = (3, 3), activation = activation, padding='same', input_shape = input_shape))
        else:
#            dilation_rate=randint(1, 4)
            model.add(Conv2D(nb_neurons, kernel_size = (3, 3), padding='same'))
            model.add(BatchNormalization())
            model.add(Activation(activation))
       # model.add(Dropout(0.2))


    model.add(Conv2D(1, (3, 3), padding='same'))


    #BAYESIAN CONVOLUTIONAL NEURAL NETWORKS WITH BERNOULLI APPROXIMATE VARIATIONAL INFERENCE
    #need to read this paper

    model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

    return model


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

def train_and_score(geneparam, dataset, u_ID, generation):
    """Train the model, return test loss.
    Args:
        network (dict): the parameters of the network
        dataset (str): Dataset to use for training/evaluating

    """
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    if not os.path.exists(args.test_dir):
        os.makedirs(args.test_dir)

    logging.info("Getting Keras datasets")

    if dataset   == 'cifar10_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_cifar10_mlp()
    elif dataset == 'cifar10_cnn':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_cifar10_cnn()
    elif dataset == 'mnist_mlp':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_mnist_mlp()
    elif dataset == 'mnist_cnn':
        nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, epochs = get_mnist_cnn()
    elif dataset == 'mnist_denoisingcnn':
        x_train, y_train, eval_data, batch_size, epochs, input_shape = get_mnist_denoisingcnn()
    elif dataset == 'CT_denoisingcnn':
        x_train, y_train, eval_data_clean, eval_data_noisy, batch_size, epochs, input_shape = get_ct_denoisingcnn()

    logging.info("Compling Keras model")

    if dataset   == 'cifar10_mlp':
        model = compile_model_mlp(geneparam, nb_classes, input_shape)
    elif dataset == 'cifar10_cnn':
        model = compile_model_cnn(geneparam, nb_classes, input_shape)
    elif dataset == 'mnist_mlp':
        model = compile_model_mlp(geneparam, nb_classes, input_shape)
    elif dataset == 'mnist_denoisingcnn':
       model = compile_model_cnn_denoising(geneparam, input_shape) 
    elif dataset == 'CT_denoisingcnn':
       model = compile_model_cnn_denoising(geneparam, input_shape) 
    elif dataset == 'mnist_cnn':
        model = compile_model_cnn(geneparam, nb_classes, input_shape)

    history = LossHistory()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              verbose=1, validation_split=0.1, callbacks=[early_stopper])
    
    #score = model.evaluate(x_test, y_test, verbose=0)
    psnr_sum = 0
    print("[*] " + 'noise level: ' + str(args.sigma) + " start testing...")
    
    if dataset == 'CT_denoisingcnn':
        for idx in range(len(eval_data_clean)):
           
            clean_image = np.array(eval_data_clean[idx])[0] 
        
            noisy_image = np.array(eval_data_noisy[idx])[0] / 255.0
            
            outputimage=np.squeeze(noisy_image)-np.squeeze(model.predict(np.array([noisy_image]))[0])
            
            outputimage = np.clip(255 * (outputimage), 0, 255).astype('uint8')
           
            groundtruth = np.squeeze(clean_image)
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(args.test_dir, 'noisy_Gen_%d_UID_%d_%d.png' % (generation,u_ID,idx)), np.array(eval_data_noisy[idx])[0])
            save_images(os.path.join(args.test_dir, 'denoised_Gen_%d_UID_%d_%d.png' % (generation,u_ID,idx)), outputimage)
    
        avg_psnr = psnr_sum / len(eval_data_clean)
    
    else:
        for idx in range(len(eval_data)):
       
            clean_image = np.array(eval_data[idx])[0] / 255.0
        
            #noisy_image = clean_image + (args.sigma / 255.0) * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)
           # noisy_image = clean_image+ tf.random_normal(shape=tf.shape(clean_image), stddev=args.sigma / 255.0)
            noisy_image = clean_image + (args.sigma / 255.0) * np.random.normal(loc=0.0, scale=1.0, size=clean_image.shape)
            
            output_clean_image=model.predict(np.array([noisy_image]))[0]
    
            groundtruth = np.clip(255 * np.squeeze(clean_image), 0, 255).astype('uint8')
            outputimage = np.clip(255 * np.squeeze(output_clean_image), 0, 255).astype('uint8')
            # calculate PSNR
            psnr = cal_psnr(groundtruth, outputimage)
            print("img%d PSNR: %.2f" % (idx, psnr))
            psnr_sum += psnr
            save_images(os.path.join(args.test_dir, 'noisy_%d_%d.png' % (u_ID,idx)), np.array(eval_data_noisy[idx])[0])
            save_images(os.path.join(args.test_dir, 'denoised_%d_%d.png' % (u_ID,idx)), outputimage)
    
        avg_psnr = psnr_sum / len(eval_data)
       
    
    print("---"+ "NetID_"+str(u_ID)+ "_GenID-" +str(generation)+"---Average PSNR %.2f ---" % avg_psnr)
    
    avg_psnr_str= "{:2.2f}".format(avg_psnr)
    modelName='NetID_'+str(u_ID)+'_GenID-'+str(generation)+'_PSNR-'+str(avg_psnr_str)+'.h5'
    model.save(os.path.join(args.ckpt_dir, modelName))
    #print('Test loss:', score[0])
    #print('Test accuracy or psnr:', score[1])

    K.clear_session()
    #we do not care about keeping any of this in memory -
    #we just need to know the final scores and the architecture

    #return score[1]  # 1 is accuracy. 0 is loss.
    return avg_psnr
