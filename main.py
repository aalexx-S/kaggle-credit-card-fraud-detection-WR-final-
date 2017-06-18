import read_data
import argparse
import sys
import sampler
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
    config_inputfile = config.get('GENERAL', 'inputfile')
    config_sample_method = config.get('SAMPLER', 'method')
    if args.input:
        config_inputfile = args.input
    if args.sample:
        config_sample_method = args.sample
        config.set('SAMPLER', 'method', config_sample_method)
    # read data
    print('Reading data.')
    X, y = read_data.read_data(config_inputfile)
    print('{0} data read.'.format(len(y)))
    # construct validation set
    print('Constructing validation data.')
    train_X, train_y, val_X, val_y\
            = valicut.valicut(X, y, float(config.get('VALIDATE', 'ratio')))
    # sample
    print('Sampling, method = {0}.'.format(config_sample_method))
    smp = sampler.Smp(config)
    train_X, train_y = smp.fit_sample(train_X, train_y)
    print('data size: {0}.'.format(len(train_y)))
    # train
    print('Training, method = {0}.'.format(config.get('TRAIN', 'method')))
    classifier = None
    if config.get('TRAIN', 'method') == 'svm':
        classifier = svm.train(train_X, train_y, config.get('SVM', 'method'))
    elif config.get('TRAIN', 'method') == 'sgd':
        class_weight = config.get('SGD', 'weight')
        # transform from str to dict
        if class_weight not in ['balanced', '']:
            class_weight = literal_eval(class_weight)
        classifier = sgd.train(train_X, train_y, class_weight,
                int(config.get('SGD', 'n_iter')))

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
    parser.add_argument('-sample')
    args = parser.parse_args()
    main(args)
