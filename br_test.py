#@Time      :2018/11/20 18:08
#@Author    :zhounan
# @FileName: br_test.py
import numpy as np
import tensorflow as tf
import evaluate_model
import random
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVC
from sklearn.externals import joblib


def predict(x_test, model_type):

    clf_list = joblib.load('./sk_model/' + model_type + '_clf_list.pkl')
    chains_order_list = joblib.load('./sk_model/' + model_type + '_chains_order_list.pkl')
    label_dim = len(clf_list)
    y = np.zeros((x_test.shape[0], label_dim))
    prob = np.zeros((x_test.shape[0], label_dim))

    if model_type == 'BR':
        for i in range(label_dim):
            y[:, i] = clf_list[i].predict(x_test)
            prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]

        return y, prob

    if model_type == 'CC':
        for i in range(label_dim):
            y[:, i] = clf_list[i].predict(x_test)
            prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]
            x_test = np.c_[x_test, y[:, i]]

        return y, prob

    if model_type == 'ECC':
        number_of_chains = len(chains_order_list)
        for i in  range(number_of_chains):
            chains_order = chains_order_list[i]
            for j in chains_order:
                y[:, j] = y[:, j] + clf_list[i][j].predict(x_test)
                x_test = np.c_[x_test, y[:, j]]
            # end for
        #end for
        y = np.around(y)
        prob = y / number_of_chains
        return y, prob

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['yeast', 'delicious']
    dataset_name = dataset_names[0]
    model_type = 'BR'
    _, _, x_test, y_test = load_data(dataset_name)
    pred, pred_prob = predict(x_test, model_type)

    print(dataset_name, model_type, 'hammingloss:', evaluate_model.hamming_loss(pred, y_test))
    print(dataset_name, 'rankingloss:', evaluate_model.rloss(pred_prob, y_test))
    print(dataset_name, 'oneerror:', evaluate_model.OneError(pred_prob, y_test))
