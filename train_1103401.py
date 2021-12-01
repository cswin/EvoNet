from keras.datasets import mnist
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dropout
from keras.callbacks import EarlyStopping, Callback
import h5py
from utils import *
import argparse
from glob import glob
from keras.layers.normalization import BatchNormalization

parser = argparse.ArgumentParser(description='')
parser.add_argument('--test_dir', dest='test_dir', default='./evalResult', help='test sample are saved here')
parser.add_argument('--eval_set_ct10clean', dest='eval_set_ct10clean', default='ct10_clean', help='dataset for eval in training')
parser.add_argument('--eval_set_ct10noisy', dest='eval_set_ct10noisy', default='ct10_noisy', help='dataset for eval in training')
args = parser.parse_args()

# Helper: Early stopping.
early_stopper = EarlyStopping( monitor='val_loss', min_delta=0.1, patience=5, verbose=0, mode='auto' )

print("**************Loading DATA**************")
#DATA

f = h5py.File('/data/TEST/DnCNN_Training/data/MICCAI_N22/imdb.mat', 'r')
y_train=np.transpose(f['labels'],(0,2,3,1))
x_train=np.transpose(f['inputs'],(0,2,3,1))
 

# Params

input_shape=[None, None, 1]
nb_layers=6
nb_neurons=64
activation='relu'
optimizer='adadelta'

print("**************Loading NETWORK**************")
# NETWORK
model = Sequential()

# Add each layer.
for i in range(0,nb_layers):
   # Need input shape for first layer.
    if i == 0:
      model.add(Conv2D(nb_neurons, kernel_size = (3, 3), activation = activation, padding='same', input_shape = input_shape))
    else:
#      dilation_rate=2  
      model.add(Conv2D(nb_neurons, kernel_size = (3, 3), activation = activation, padding='same', dilation_rate=2))
      model.add(BatchNormalization())

#    model.add(Dropout(0.2))


model.add(Conv2D(1, (3, 3), padding='same'))


model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

 
model.compile(optimizer=optimizer, loss='binary_crossentropy')

model.fit(x_train, y_train,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_split=0.2,callbacks=[early_stopper]
             )

print("Saving trained model")
model.save("1103401.h5")

print("*****************TESTING model:**********************")
# test data during training
eval_files_clean = glob('./data/test/ct10_clean/*.bmp'.format(args.eval_set_ct10clean))
eval_data_clean = load_images(eval_files_clean) 
eval_files_noisy = glob('./data/test/ct10_noisy_20mA/*.bmp'.format(args.eval_set_ct10noisy))
eval_data_noisy = load_images(eval_files_noisy) 
psnr_sum = 0
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

    save_images(os.path.join(args.test_dir, 'noisy_%d.png' % (idx)), np.array(eval_data_noisy[idx])[0])
    save_images(os.path.join(args.test_dir, 'denoised_%d.png' % (idx)), outputimage)
    
avg_psnr = psnr_sum / len(eval_data_clean)
print("Average PSNR: %.2f" % (avg_psnr))
 
 
 
 
 
 
 
