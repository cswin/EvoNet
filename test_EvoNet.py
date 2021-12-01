from keras.datasets import mnist
import numpy as np
from keras.models import load_model
from keras.layers import Conv2D, Dropout
from keras.callbacks import EarlyStopping, Callback
import h5py
from utils import *
import argparse
from glob import glob


parser = argparse.ArgumentParser(description='')
parser.add_argument('--test_dir', dest='test_dir', default='./evalResult', help='test sample are saved here')
parser.add_argument('--eval_set_ct10clean', dest='eval_set_ct10clean', default='ct10_clean', help='dataset for eval in training')
parser.add_argument('--eval_set_ct10noisy', dest='eval_set_ct10noisy', default='ct10_noisy', help='dataset for eval in training')
args = parser.parse_args()



model=load_model("1103401.h5")


 
 
eval_files_clean = glob('./data/test/ct10_clean/*.bmp'.format(args.eval_set_ct10clean))
eval_data_clean = load_images(eval_files_clean) 
eval_files_noisy = glob('./data/test/ct10_noisy_20mA/*.bmp'.format(args.eval_set_ct10noisy))
eval_data_noisy = load_images(eval_files_noisy) 

psnr_sum = 0
for idx in range(len(eval_data_clean)):
           
    clean_image = np.array(eval_data_clean[idx])[0] 
        
    noisy_image = np.array(eval_data_noisy[idx])[0] / 255.0
            
    outputimage=model.predict(np.array([noisy_image]))[0]
    outputimage = np.clip(255 * np.squeeze(outputimage), 0, 255).astype('uint8')
            
    groundtruth = np.squeeze(clean_image)
    # calculate PSNR
    psnr = cal_psnr(groundtruth, outputimage)
    print("img%d PSNR: %.2f" % (idx, psnr))
    psnr_sum += psnr
    save_images(os.path.join(args.test_dir, 'noisy_%d.png' % (idx)), np.array(eval_data_noisy[idx])[0])
    save_images(os.path.join(args.test_dir, 'denoised_%d.png' % (idx)), outputimage)
    
avg_psnr = psnr_sum / len(eval_data_clean)
print("Average PSNR: %.2f" % (avg_psnr))