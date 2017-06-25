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
    Utils.verbose_print('Reading data.')
    X, y = Utils.read_data(config.get('GENERAL', 'inputfile'))
    Utils.verbose_print('{0} data read.'.format(len(y)))

    # construct validation set
    Utils.verbose_print('Constructing validation data.')
    train_X, train_y, val_X, val_y\
            = Utils.valicut(X, y, float(config.get('VALIDATE', 'ratio')))
    val_X_amount = [i[-1] for i in val_X]

    # feature select
    selector = Utils.get_feature_selector(config)
    if selector:
        selector.fit(train_X, train_y)
        before_trans_0 = train_X[0]
        train_X = selector.transform(train_X)
        val_X = selector.transform(val_X)
        Utils.verbose_print('pvalues:' + str(selector.pvalues_))
        Utils.verbose_print('{0} features left: {1}'.format(len(train_X[0]),
                            Utils.get_index(before_trans_0, train_X[0])))
    # standardization
    Utils.verbose_print('Standardizing.')
    scaler = StandardScaler().fit(train_X)
    train_X = scaler.transform(train_X)
    val_X = scaler.transform(val_X)

    # sample
    Utils.verbose_print('Sampling, method = {0}.'.format(config.get('SAMPLER', 'method')))
    smp = sampler.Smp(config)
    train_X, train_y = smp.fit_sample(train_X, train_y)
    Utils.verbose_print('data size: {0}.'.format(len(train_y)))

    # train
    Utils.verbose_print('Training, method = {0}.'.format(config.get('CLASSIFIER', 'method')))
    clf = classifier.Clf(config)
    clf.fit(train_X, train_y)

    # validate
    Utils.verbose_print('Validating.')
    result_y = clf.predict(val_X)
    correction = Utils.correction(val_y, result_y)
    truth_table = Utils.truth_table(val_y, result_y)
    amount_cost = Utils.EDCS_cost(val_y, result_y, val_X_amount)
    print('Correction:{0}'.format(correction))
    Utils.verbose_print(Utils.print_truth_table(truth_table))
    f1_score, Precision, Recall = Utils.f1_score(truth_table)
    Utils.verbose_print('F1 Score:{0}'.format(f1_score))
    Utils.verbose_print('Precision:{0}'.format(Precision))
    Utils.verbose_print('Recall:{0}'.format(Recall))
    Utils.verbose_print('Amount sensitive cost:{0}'.format(amount_cost))


if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-i', '--input', nargs='?')
    parser.add_argument('-s', '--sample', nargs='?')
    parser.add_argument('-v', '--verbose', action='store_true')
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
    if args.verbose:
        Utils.set_verbose()

    main(config)
