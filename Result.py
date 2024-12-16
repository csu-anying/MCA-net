import numpy as np
from sklearn.metrics import *

def accuracy_subset(y_pred, y_true, threash = 0.5):
    y_pred = np.where(y_pred > threash, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def ACC(con_mat, n = 2):
    acc = []
    number = np.sum(con_mat[:,:])
    temp = 0
    for i in range(n):
        temp += con_mat[i][i]
    acc = temp / number    
    return acc

def location_Result(class_num,y_pred,y_test):
    output_labels = []
    
    for i,key in enumerate(y_pred):
        output_label = []
        
        for j in range(len(key)):
            if(key[j] >= 0.5):
                output_label.append(1)
            else:
                output_label.append(0)
        
        output_label = np.array(output_label)
        output_labels.append(output_label)
    
    output_labels = np.array(output_labels)
    
    output = output_labels
    label = y_test

    ACC = accuracy_subset(output,label)
    
    # if (int(class_num) == 5):
    #     print("Location-5-ACC：",(ACC))
    # elif (int(class_num) == 7):
    #     print("Location-7-ACC：",(ACC))

    return ACC

def detection_Result(y_pred,y_test):

    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test,axis=1)
    
    con_mat = confusion_matrix(y_test, y_pred)
    n = con_mat.shape[0]
    acc = ACC(con_mat,n)
    # print('Detection_ACC: ',acc)

    return acc

def Pre_Result(d, l_5, l_7, ret, threash = 0.5):
    d_label = ["MI", "NORM"]
    l5_label = ['AMI','ASMI','ALMI','other','NORM']
    l7_label = ['AMI','ASMI','ALMI','IMI','ILMI','other','NORM']

    max_value = max(d)
    max_idx = d.index(max_value)
    ret += 'Detection: {}\n'.format(d_label[max_idx]) 
    # print("Detection: {}".format(d_label[max_idx]))
    
    loc_5 = np.array(l_5)
    loc_7 = np.array(l_7)

    loc_5 = np.where(loc_5 > threash, 1, 0).tolist()
    loc_7 = np.where(loc_7 > threash, 1, 0).tolist()

    pre_5 = ""
    pre_7 = ""

    for i in range(5):
        if (loc_5[i] == 1):
            pre_5 += l5_label[i] + " "
    
    ret += 'Location-5: {}\n'.format(pre_5) 
    # print("Location-5: {}".format(pre_5))

    for i in range(7):
        if (loc_7[i] == 1):
            pre_7 += l7_label[i] + " "
    
    ret += 'Location-7: {}\n'.format(pre_7) 
    # print("Location-7: {}".format(pre_7))

    return ret
    