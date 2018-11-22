#@Time      :2018/11/20 18:07
#@Author    :zhounan
# @FileName: br_train.py
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.externals import joblib
import evaluate_model

def train(x_train, y_train, model_type):
    data_num = x_train.shape[0]
    feature_dim = x_train.shape[1]
    label_dim = y_train.shape[1]

    clf_list = []
    if model_type == 'BR':
        for i in range(label_dim):
            clf = SVC(kernel='rbf', probability=True)
            clf.fit(x_train, y_train[:,i].ravel())
            clf_list.append(clf)
        #end for

        joblib.dump(clf_list, './sk_model/' + model_type + '_clf_list.pkl')

    if model_type == 'CC':
        for i in range(label_dim):
            clf = SVC(kernel='rbf', probability=True)
            clf.fit(x_train, y_train[:,i].ravel())
            pred = clf.predict(x_train)
            x_train = np.c_[x_train, pred]
            clf_list.append(clf)
        #end for

        joblib.dump(clf_list, './sk_model/' + model_type + '_clf_list.pkl')

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['yeast','delicious']
    dataset_name = dataset_names[0]
    model_type = 'BR'
    x_train, y_train, _, _ = load_data(dataset_name)
    train(x_train, y_train, model_type)