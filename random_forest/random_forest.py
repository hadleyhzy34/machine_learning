import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from decision_tree import build_tree
from decision_tree import print_tree
from decision_tree import classify


def encode_data(training_data):
    #input must be list, output is list
    n_features = len(training_data[0])
    data_set = []
    label_encoder = LabelEncoder()
    for col in range(n_features):
        x = [ data[col] for data in training_data ]
        if isinstance(training_data[0][col],int):
            data_set.append(x)
            continue
        y = label_encoder.fit_transform(x)
        temp = y.tolist()
        data_set.append(temp)
    arr = np.array(data_set)
    arr = arr.T
    data_set = arr.tolist()
    return data_set

def training_model(training_data, estimators):
    #bagging
    _size = len(training_data)
    container = []
    for i in range(estimators):
        sample = random.sample(training_data, int(_size/5))
        container.append(build_tree(sample))

    return container


def predict(testing_data, container, estimators):
    right = 0
    wrong = 0
    for data in testing_data:
        one = 0
        zero = 0
        for i in range(estimators):
            if classify(data, container[i]) == 1:
                one += 1
            else:
                zero += 1
        if one > zero:
            res = one
        else:
            res = zero
        
        if res == data[-1]:
            right += 1
        else:
            wrong += 1
    return right, wrong


