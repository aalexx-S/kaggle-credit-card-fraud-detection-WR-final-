"""
2017 web retrieval final project.
See https://github.com/aalexx-S/WR-final
"""
import argparse
import sys
import Utils
import numpy as np
from ast import literal_eval
from sklearn.preprocessing import StandardScaler

import sampler
import configparser
import classifier

def main(config):
    # read data
    print('Reading data.')
    X, y = Utils.read_data(config.get('GENERAL', 'inputfile'))
    print('{0} data read.'.format(len(y)))
    # construct validation set
    print('Constructing validation data.')
    train_X, train_y, val_X, val_y\
            = Utils.valicut(X, y, float(config.get('VALIDATE', 'ratio')))
    # standardization
    print('Standardizing.')
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)
    # sample
    print('Sampling, method = {0}.'.format(config.get('SAMPLER', 'method')))
    smp = sampler.Smp(config)
    train_X, train_y = smp.fit_sample(train_X, train_y)
    print('data size: {0}.'.format(len(train_y)))
    # train
    print('Training, method = {0}.'.format(config.get('CLASSIFIER', 'method')))
    clf = classifier.Clf(config)
    clf.fit(train_X, train_y)
    # validate
    print('Validating.')
    result_y = clf.predict(val_X)
    correction = Utils.correction(val_y, result_y)
    truth_table = Utils.truth_table(val_y, result_y)
    print('Correction:{0}'.format(correction))
    Utils.print_truth_table(truth_table)


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-input')
    parser.add_argument('-sample')
    args = parser.parse_args()
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_inputfile = config.get('GENERAL', 'inputfile')
    config_sample_method = config.get('SAMPLER', 'method')
    if args.input:
        config.set('GENERAL', 'inputfile', args.input)
    if args.sample:
        config.set('SAMPLER', 'method', args.sample)
    main(config)
