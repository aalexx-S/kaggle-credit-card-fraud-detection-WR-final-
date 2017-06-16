import read_data
import argparse
import sys
import oversample
import configparser
import valicut
import numpy as np

def main(args):
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_inputfile = config['GENERAL']['inputfile']
    config_oversamplemode = config['OVERSAMPLE']['mode']
    if args.input:
        config_inputfile = args.input
    if args.oversample:
        config_oversamplemode = args.oversample
    # read data
    X, y = read_data.read_data(config_inputfile)
    # construct validation set
    train_X, train_y, val_X, val_y\
            = valicut.valicut(X, y, float(config['VALIDATE']['ratio']))
    # oversample
    if config_oversamplemode == 'SMOTE':
        train_X, train_y = oversample.over_sampling_SMOTE_imblearn(
                train_X, train_y,
                config['OVERSAMPLE']['kind'])
    elif config_oversamplemode == 'naive':
        train_X, train_y = oversample.over_sampling_naive(
                train_X, train_y,
                config['OVERSAMPLE']['ratio'])
    # train


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-input')
    parser.add_argument('-oversample')
    args = parser.parse_args()
    main(args)
