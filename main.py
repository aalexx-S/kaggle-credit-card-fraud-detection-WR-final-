import read_data
import argparse
import sys
import oversample
import configparser
import valicut
import numpy as np
import svm
import validate
import sgd
from ast import literal_eval

def main(args):
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_inputfile = config['GENERAL']['inputfile']
    config_oversample_method = config['OVERSAMPLE']['method']
    if args.input:
        config_inputfile = args.input
    if args.oversample:
        config_oversample_method = args.oversample
    # read data
    print('Reading data.')
    X, y = read_data.read_data(config_inputfile)
    print('{0} data read.'.format(len(y)))
    # construct validation set
    print('Constructing validation data.')
    train_X, train_y, val_X, val_y\
            = valicut.valicut(X, y, float(config['VALIDATE']['ratio']))
    # oversample
    print('Oversampling, method = {0}.'.format(config_oversample_method))
    if config_oversample_method == 'SMOTE':
        train_X, train_y = oversample.over_sampling_SMOTE_imblearn(
                train_X, train_y,
                config['OVERSAMPLE']['kind'])
    elif config_oversample_method == 'naive':
        train_X, train_y = oversample.over_sampling_naive(
                train_X, train_y,
                float(config['OVERSAMPLE']['ratio']))
    # train
    print('Training, method = {0}.'.format(config['TRAIN']['method']))
    classifier = None
    if config['TRAIN']['method'] == 'svm':
        classifier = svm.train(train_X, train_y, config['SVM']['method'])
    elif config['TRAIN']['method'] == 'sgd':
        class_weight = config['SGD']['weight']
        # transform from str to dict
        if class_weight not in ['balanced', '']:
            class_weight = literal_eval(class_weight)
        classifier = sgd.train(train_X, train_y, class_weight,
                int(config['SGD']['n_iter']))

    # validate
    print('Validating.')
    result_y = classifier.predict(val_X)
    correction = validate.correction(val_y, result_y)
    truth_table = validate.truth_table(val_y, result_y)
    print('Correction:{0}'.format(correction))
    validate.print_truth_table(truth_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-input')
    parser.add_argument('-oversample')
    args = parser.parse_args()
    main(args)
