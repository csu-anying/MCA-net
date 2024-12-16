import numpy as np
import tensorflow as tf
import argparse
import sys
import os
from tensorflow import keras
from tensorflow.keras.models import *

from keras import backend as K
from MTMLSEnet import mtmlsenet
from Result import *

def Pre_Record(test, work_path):
    model = mtmlsenet([5000,1],1)

    model.compile(loss={'detection':'binary_crossentropy','location_5':'binary_crossentropy','location_7':'binary_crossentropy'},
        loss_weights={'detection':1.,'location_5':15.,'location_7':60.}, 
        optimizer='adam', 
        metrics=['accuracy'])
    
    K.set_value(model.optimizer.lr,0.001)
    
    det = []
    loc_5 = []
    loc_7 = []

    for i in range(10):
        filepath = os.path.join(work_path, './saved_models/fold{}.weights.best.hdf5'.format(i+1))
        model.load_weights(filepath)
        y_pred_detection_1, y_pred_location5_1, y_pred_location7_1 = model.predict(test)
        
        if (len(det) != 0) :
            det = np.sum([det,y_pred_detection_1],axis=0).tolist()
        else :
            det = y_pred_detection_1
        
        if (len(loc_5) != 0) :
            loc_5 = np.sum([loc_5,y_pred_location5_1],axis=0).tolist()
        else :
            loc_5 = y_pred_location5_1

        if (len(loc_7) != 0) :
            loc_7 = np.sum([loc_7,y_pred_location7_1],axis=0).tolist()
        else :
            loc_7 = y_pred_location7_1

    return det, loc_5, loc_7

def check_and_pred(ret, work_path):

    # load pred_data
    path = os.path.join(work_path, 'data/user_predict')

    def getFlist(path):
        for root, dirs, files in os.walk(path):
            z = []
        return files

    if not os.path.exists(path):
        ret += 'ERROR! The data path is incorrect. Please put the data in the specified path.\n'
        return ret

    file_list = getFlist(path)

    for file in file_list:
        a = np.load(os.path.join(path, file))
        shape = a.shape
        slen = len(shape)
        test = a
        
        ret += 'The Myocardial Infarction Predictions for patient %s are as follows:\n' % file[:-4]
        # print("Record Name: {}".format(file))

        def re_shape(X):
            x=[]
            for i in range(12):
                x.append(X[:,:,i:i+1])
            return x

        # check input_dim and input_shape and precessing 
        if (file[-4:] != ".npy"):
            ret += 'File format is error! Expected file format: \".npy\"\n\n'
            # print("Expected file format: \".npy\".")
            continue
            
        if (slen != 2):
            ret += 'Expected input_dim = 2, found input_dim = {}.\n\n'.format(slen)
            # print("Expected input_dim = 2, found input_dim = {}.".format(slen))
            continue
        
        if (shape[0] == 12 and shape[1] == 5000):
            test = np.expand_dims(a, axis = 0)
            test = test.transpose(0,2,1)
            test = re_shape(test)

        elif (shape[0] == 5000 and shape[1] == 12):
            test = np.expand_dims(a, axis = 0)
            test = re_shape(test)

        else :
            ret += 'Expected input_shape = (5000, 12) or (12, 5000) found input_shape = {}.\n\n'.format(shape)
            # print("Expected input_shape = (5000, 12) or (12, 5000) found input_shape = {}.".format(shape))
            continue
        
        det, loc_5, loc_7 = Pre_Record(test, work_path)

        d = [a/10 for a in det[0]]
        l_5 = [a/10 for a in loc_5[0]]
        l_7 = [a/10 for a in loc_7[0]]

        ret = Pre_Result(d, l_5, l_7, ret)
        ret += '\n'

    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='configuraitons')
    parser.add_argument('--work_path', type=str, default='.', help='work path of workflow')
    args = parser.parse_args()
    work_path = args.work_path

    # Parse arguments.
    # if len(sys.argv) != 1:
    #     raise Exception('e.g., CUDA_VISIBLE_DEVICES=? python check_and_pred.py')

    # Run the training code.
    ret = ''
    ret += 'Running check and predict code...\n\n'
    ret = check_and_pred(ret, work_path)
    ret += 'Done.'

    print(ret)

    if not os.path.exists(os.path.join(work_path, 'result')):
        os.mkdir(os.path.join(work_path, 'result'))
    log_file = os.path.join(work_path, 'result/user_predict_results.txt') 
    with open(log_file, 'w') as f:
        f.write(ret + '\n')
