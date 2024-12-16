#!/usr/bin/env python
import numpy as np, os, sys, argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from keras import backend as K
from MTMLSEnet import mtmlsenet
from Result import *


def test(ret, work_path):

    det = []
    loc_5 = []
    loc_7 = []

    for i in range(10):
        print('KFold:%d'%(i+1))
    
        X_test_detection_1 = np.load(os.path.join(work_path, "./data/valid_evaluation/X_test_f{}.npy".format(i)))
        y_test_detection_1 = np.load(os.path.join(work_path, "./data/valid_evaluation/y_test_detection_f{}.npy".format(i)))
        y_test_location5_1 = np.load(os.path.join(work_path, "./data/valid_evaluation/y_test_location5_f{}.npy".format(i)))
        y_test_location7_1 = np.load(os.path.join(work_path, "./data/valid_evaluation/y_test_location7_f{}.npy".format(i)))
        # (12, 1491, 5000, 1)
        # print(X_test_detection_1[0].shape)
        # print(X_test_detection_1[0])

        X_test_d = []
        for z in X_test_detection_1:
            # print(z.shape)
            X_test_d.append(z)
            # print(np.array(X_test_d).shape)

        X_test_detection_1 = X_test_d


        model = mtmlsenet([5000,1],1)

        model.compile(loss={'detection':'binary_crossentropy','location_5':'binary_crossentropy','location_7':'binary_crossentropy'},
            loss_weights={'detection':1.,'location_5':15.,'location_7':60.}, 
            optimizer='adam', 
            metrics=['accuracy'])
        
        K.set_value(model.optimizer.lr,0.001)

        filepath = os.path.join(work_path, './saved_models/fold{}.weights.best.hdf5'.format(i+1))

        model.load_weights(filepath)
        y_pred_detection_1, y_pred_location5_1, y_pred_location7_1 = model.predict(X_test_detection_1)

        d = detection_Result(y_pred_detection_1,y_test_detection_1)
        # print(d)
        
        l_5 = location_Result(5,y_pred_location5_1,y_test_location5_1)
        # print(l_5)

        l_7 = location_Result(7,y_pred_location7_1,y_test_location7_1)
        # print(l_7)
        
        det.append(d)
        loc_5.append(l_5)
        loc_7.append(l_7)

    ret += 'The evaluation of the model are as follows: \n'
    ret += 'detection acc = {}\n'.format(sum(det)/len(det))
    ret += 'location-5 acc = {}\n'.format(sum(loc_5)/len(loc_5))
    ret += 'location-7 acc = {}\n'.format(sum(loc_7)/len(loc_7))
    ret += '\n'
    return ret

if __name__ == '__main__':
    # Parse arguments.
    # if len(sys.argv) != 1:
    #     raise Exception('e.g., CUDA_VISIBLE_DEVICES=? python test.py')

    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--work_path', type=str, default='.', help='work path of workflow')
    args = parser.parse_args()
    work_path = args.work_path

    # Run the training code.
    ret = ''
    ret += 'Running testing code...\n'
    ret = test(ret, work_path)

    print(ret)

    if not os.path.exists(os.path.join(work_path, 'result')):
        os.mkdir(os.path.join(work_path, 'result'))
    log_file = os.path.join(work_path, 'result/valid_evaluation.txt') 
    with open(log_file, 'w') as f:
        f.write(ret + '\n')