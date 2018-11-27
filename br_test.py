#@Time      :2018/11/20 18:08
#@Author    :zhounan
# @FileName: br_test.py
import numpy as np
import evaluate_model
import argparse
from sklearn.svm import SVC
from sklearn.externals import joblib
def get_args():
    ## hyperparameters
    parser = argparse.ArgumentParser(description='neural networks for multilabel learning')
    parser.add_argument('--dataset_name', type=str, default='yeast', help='train data name')
    parser.add_argument('--label_dim', type=int, default=0, help='#sample of each minibatch')
    parser.add_argument('--data_num', type=int, default=0, help='#epoch of training')
    parser.add_argument('--model_type', type=str, default='CC', help='model type(BR,CC,ECC,PCC)')
    args = parser.parse_args()

    return args

def predict(x_test, args):
    clf_list = joblib.load('./sk_model/' + args.model_type + '_clf_list.pkl')
    if args.model_type == 'ECC':
        args.label_dim = len(clf_list[0])
        chains_order_list = joblib.load('./sk_model/' + args.model_type + '_chains_order_list.pkl')
    else:
        args.label_dim = len(clf_list)
    y = np.zeros((x_test.shape[0], args.label_dim))
    prob = np.zeros((x_test.shape[0], args.label_dim))

    if args.model_type == 'BR':
        for i in range(args.label_dim):
            y[:, i] = clf_list[i].predict(x_test)
            prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]

        return y, prob

    if args.model_type == 'CC':
        for i in range(args.label_dim):
            y[:, i] = clf_list[i].predict(x_test)
            prob[:, i] = clf_list[i].predict_proba(x_test)[:, 1]
            x_test = np.c_[x_test, y[:, i]]

        return y, prob

    if args.model_type == 'ECC':
        number_of_chains = len(chains_order_list)
        y_ensemble = np.zeros((number_of_chains, x_test.shape[0], args.label_dim))
        for i in  range(number_of_chains):
            chains_order = chains_order_list[i]
            for j in range(len(chains_order)):
                y_ensemble[i, :, chains_order[j]] = clf_list[i][j].predict(x_test)
                x_test = np.c_[x_test, y_ensemble[i, :, chains_order[j]]]
            # end for
        #end for
        y = np.around(np.mean(y_ensemble, axis=0))
        prob = np.mean(y_ensemble, axis=0)
        return y, prob

    if args.model_type == 'PCC':
        data_num = x_test.shape[0]
        y = np.zeros((data_num, args.label_dim))
        prob = np.zeros(data_num)

        for i in range(2 ** args.label_dim):
            if i % 50 == 0:
                print(i)

            p = np.zeros((data_num, args.label_dim))
            y_bin = np.array(list(map(int, np.binary_repr(i, width=args.label_dim))))
            x_test_temp = x_test
            for j in range(args.label_dim):
                p[:, j] = clf_list[j].predict_proba(x_test_temp )[:, 1]
                x_test_temp  = np.c_[x_test_temp , np.ones(data_num).reshape(-1, 1) * y_bin[j]]

            temp_p = np.prod(p, axis=1)
            for j in range(data_num):
                if temp_p[j] > prob[j]:
                    prob[j] = temp_p[j]
                    y[j] = y_bin
        #end for
        return y, prob.reshape(-1, 1)

def load_data(dataset_name):
    x_train = np.load('./dataset/' + dataset_name + '/x_train.npy')
    y_train = np.load('./dataset/' + dataset_name + '/y_train.npy')
    x_test = np.load('./dataset/' + dataset_name + '/x_test.npy')
    y_test = np.load('./dataset/' + dataset_name + '/y_test.npy')

    return x_train, y_train, x_test, y_test

if __name__ == '__main__':
    dataset_names = ['yeast', 'delicious']
    args = get_args()
    dataset_name = dataset_names[0]
    args.dataset_name = dataset_name
    args.model_type = 'PCC'
    _, _, x_test, y_test = load_data(args.dataset_name)
    pred, pred_prob = predict(x_test, args)

    print(dataset_name, args.model_type, 'hammingloss:', evaluate_model.hamming_loss(pred, y_test))
    print(dataset_name, 'rankingloss:', evaluate_model.rloss(pred_prob, y_test))
    print(dataset_name, 'oneerror:', evaluate_model.OneError(pred_prob, y_test))
