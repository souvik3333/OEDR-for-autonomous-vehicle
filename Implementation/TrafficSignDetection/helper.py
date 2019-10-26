import pickle
import sys
import argparse
import numpy as np
import cv2

parser=argparse.ArgumentParser()
parser.add_argument('--training_file', help='path to train file',default="train.p")
parser.add_argument('--testing_file', help='path to test file',default="valid.p")
parser.add_argument('--train_batch_size', help='train batch size',default=16)
parser.add_argument('--test_batch_size', help='test batch size',default=1000)

def reshape_raw_images(imgs):
    """Given 4D images (number, heigh, weight, channel), this
    function grayscales and returns (number, height, weight, 1) images"""
    def gray(src):
        if src.dtype == np.uint8:
            src = np.array(src/255.0, dtype=np.float32)
        dst = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        return dst.reshape(32,32,1)
    norms = [gray(img) for img in imgs]
    return np.array(norms)

def load_dataset(training_file,testing_file):
    """
        loads datasets
        arguments:  
                    training_file : training data file location path
                    testing_file : test data file location path
    """
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    with open(validation_file, mode='rb') as f:
        valid = pickle.load(f)
    X_train, y_train = train['features'], train['labels']
    X_valid, y_valid = valid['features'], valid['labels']
    return (X_train,y_train,X_valid,y_valid)

def read_args(train_batch_size,test_batch_size):
    args=parser.parse_args()
    print(args.training_file)

    return (training_file,testing_file,train_batch_size,test_batch_size)