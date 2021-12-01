import numpy as np
#from keras.datasets  import mnist
import scipy.io as sio
import scipy.ndimage
import PIL as Image
import matplotlib.pylab as plt
from scipy.ndimage import gaussian_filter
import h5py


f = h5py.File('./data/imdb_CT35_N22_LabelClean_noaug.mat', 'r')
y_train_clean=np.transpose(f['labels'],(0,2,3,1))
 
 

#trainset=np.load('./data/img_clean_pats.npy')
#im=np.array(trainset)[1,]
#X,Y,T=im.shape
#im=im.astype('float32') / 255.
##x_train = trainset.astype('float32') / 255.
##x_train_noisy = x_train + 0.5 * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
##
##(x_train_mnist, _), (x_test_mnist, _) =mnist.load_data()
#
#mat_contents = sio.loadmat('acf.mat')
#acf=mat_contents['acf']
#
#sigma=25
#im=np.squeeze(im)
#noise=np.random.randn(X,Y)
#noise=scipy.ndimage.convolve(im, acf, mode='mirror')
##noise = Image.imfilter(noise,acf,'symmetric','conv');
##noise = sigma/255.
##
##
#noise=noise * ('noise - np.mean(noise))/np.std(noise)
#
#noisy_image= np.array(im) + np.array(noise)
#
#
#noisy_image=np.array((noisy_image*255),dtype=np.uint8)
#im=Image.fromarray(np.squeeze(noisy_image))
#im.save('tmp_noise.png')

#eng = matlab.engine.start_matlab()
#eng.edit('pct_noise_simple',nargout=0)
#noisy, noise=eng.pct_noise_simple(im,sigma)  
