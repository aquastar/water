import csv
import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD, Adam
# Generate dummy data
from keras.regularizers import l2, l1, l1_l2


def diff(first, second):
    second = set(second)
    return [item for item in first if item not in second]


def homo(data):
    training_set = []
    test_set = []
    res = defaultdict(list)
    for _ in data:
        res[_[0]].append(_.astype(np.float))
    for k, v in res.items():
        v_len = len(v)
        test_index = random.sample(xrange(v_len), v_len / 10)
        training_index = diff(xrange(v_len), test_index)
        training_set.extend(np.array(v)[training_index].tolist())
        test_set.extend(np.array(v)[test_index].tolist())

    return np.array(training_set)[:, -1:], np.array(training_set)[:, 2:-1], np.array(
        test_set)[:, -1:], np.array(test_set)[:, 2:-1]


def heter(data):
    training_set = []
    test_set = []

    v_len = len(data)
    data = data.astype(np.float)
    training_index = random.sample(xrange(v_len), v_len / 10)
    test_index = diff(xrange(v_len), training_index)
    training_set = data[training_index]
    test_set = data[test_index]
    return np.array(training_set)[:, -1:], np.array(training_set)[:, 2:-1], np.array(
        test_set)[:, -1:], np.array(test_set)[:, 2:-1]


def sim(data):
    data = np.load('sim.npy')

    training_set = []
    test_set = []

    v_len = len(data)
    data = data.astype(np.float)
    training_index = random.sample(xrange(v_len), v_len / 10)
    test_index = diff(xrange(v_len), training_index)
    training_set = data[training_index]
    test_set = data[test_index]
    return np.array(training_set)[:, -1:], np.array(training_set)[:, :-1], np.array(
        test_set)[:, -1:], np.array(test_set)[:, :-1]


def get_whole_rst(pred, real):
    nde = np.mean(np.sqrt(np.power(np.subtract(pred.flatten(), real.flatten()), 2) / np.power(
        np.fmax(pred.flatten(), real.flatten()) + 0.00001, 2)))
    avg_acc = 1 - np.mean(np.abs(np.subtract(pred.flatten(), real.flatten())) / (
        np.fmax(pred.flatten(), real.flatten()) + 0.00001))
    recall = np.mean(np.fmin(pred.flatten(), real.flatten()) / (
        np.fmax(pred.flatten(), real.flatten()) + 0.00001))
    f1 = 2 * (avg_acc * recall) / (avg_acc + recall)

    return avg_acc, recall, f1, nde,


def get_appl_rst(pred, real, col=0):
    pred = pred[:, col]
    real = real[:, col]
    nde = np.mean(np.sqrt(np.power(np.subtract(pred.flatten(), real.flatten()), 2) / np.power(
        np.fmax(pred.flatten(), real.flatten()) + 0.00001, 2)))
    avg_acc = 1 - np.mean(np.abs(np.subtract(pred.flatten(), real.flatten())) / (
        np.fmax(pred.flatten(), real.flatten()) + 0.00001))
    recall = np.mean(np.fmin(pred.flatten(), real.flatten()) / (
        np.fmax(pred.flatten(), real.flatten()) + 0.00001))
    f1 = 2 * (avg_acc * recall) / (avg_acc + recall)

    return avg_acc, recall, f1, nde,


def metric_var(metric):
    metric = np.array(metric)
    metric_list = ['avg_acc', 'recall', 'f1', 'nde']
    for _ in xrange(len(metric_list)):
        print metric_list[_], ':{0:.4f}$\pm${1:.4f}'.format(np.mean(metric[:, _]), np.std(metric[:, _]))


def metric_var_device(metric):
    metric = np.array(metric)
    appl_list = ['faucet', 'dishwasher', 'toilet', 'shower', 'clothes']
    mtx = []
    for _ in xrange(5):
        print appl_list[_]
        metric_list = ['avg_acc', 'recall', 'f1', 'nde']
        one_device = []
        for __ in xrange(len(metric_list)):
            print metric_list[__], ':{0:.4f}$\pm${1:.4f}'.format(np.mean(metric[_, :, __]), np.std(metric[_, :, __]))
            one_device.append('{0:.4f}$\pm${1:.4f}'.format(np.mean(metric[_, :, __]), np.std(metric[_, :, __])))
        mtx.append(one_device)
    mtx = np.array(mtx).transpose().tolist()
    for _ in mtx:
        print '&' + '&'.join(_) + '\\\\'


def dense_model(method='l1'):
    model = Sequential()
    if method == 'l1':
        model.add(Dense(64, activation='relu', input_dim=1))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1(), bias_regularizer=l1()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1(), bias_regularizer=l1()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1(), bias_regularizer=l1()))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu'))
    elif method == 'l2':
        model.add(Dense(64, activation='relu', input_dim=1))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l2(), bias_regularizer=l2()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l2(), bias_regularizer=l2()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l2(), bias_regularizer=l2()))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu'))
    elif method == 'l1l2':
        model.add(Dense(64, activation='relu', input_dim=1))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='ones',
                        kernel_regularizer=l1_l2(), bias_regularizer=l1_l2()))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu'))
    else:
        model.add(Dense(64, activation='relu', input_dim=1))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation='relu'))

    return model


if __name__ == '__main__':

    all_data = []
    with open('data.csv', 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            tmp = row[1].split()[0].split('/')
            row[1] = (int(tmp[2]) - 1995) * 10000 + int(tmp[0]) * 100 + int(tmp[1])
            # appl_list = ['clothes', 'dishwasher', 'faucet', 'shower', 'toilet']

            all_data.append([row[0], row[1], row[6], row[5], row[13], row[11], row[3], row[16]])

    data_size = len(all_data)
    fold_n = 10
    fold_size = data_size / fold_n

    all_data = np.array(all_data)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    acc_list = []
    for _d in [ 'homo', 'heter']:
        if _d == 'homo':
            model_func = homo
        elif _d == 'heter':
            model_func = heter
        elif _d == 'sim':
            model_func = sim

        x_train, y_train, x_test, y_test = model_func(all_data)
        # acc/recall/f1/nde
        whole_metric = []
        appli_metric = [[], [], [], [], []]

        for _con in ['l1', 'l2', 'dense']:
            print '########################################'
            print '- Method :', _d, _con
            model = dense_model(method=_con)
            model.compile(loss='mean_squared_logarithmic_error', optimizer=sgd, metrics=['accuracy'])
            for _ in xrange(10):
                model.fit(x_train, y_train, epochs=20, batch_size=512, verbose=0)

                # Testing
                inp = model.input  # input placeholder
                outputs = [layer.output for layer in model.layers]  # all layer outputs
                functor = K.function([inp] + [K.learning_phase()], outputs)  # evaluation function
                layer_outs = functor([x_test, 0.])[-1]

                whole_metric.append(list(get_whole_rst(layer_outs, y_test)))
                for _app in xrange(5):
                    appli_metric[_app].append(list(get_appl_rst(layer_outs, y_test, col=_app)))

            print '- Whole Home'
            metric_var(whole_metric)
            print '- Device-wise'
            metric_var_device(appli_metric)
