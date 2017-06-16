import read_data
import argparse
import sys

def main(input_file):
    X, y = read_data.read_data(input_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='WR final program.')
    parser.add_argument('-input')
    args = parser.parse_args()
    main(args.input)
