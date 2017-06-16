import read_data
import argparse
import sys
import oversample
import configparser
import valicut

def main(input_file):
    # read config
    config = configparser.ConfigParser()
    config.read('config.ini')
    X, y = read_data.read_data(input_file)
    train_X, train_y, val_X, val_y\
            = valicut.valicut(float(config['VALIDATE']['ratio']), X, y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-input', required=True)
    args = parser.parse_args()
    main(args.input)
