import configparser
import numpy as np
import sys

from main import main as mainprog


class PrintMagic:
    def __init__(self):
        self.buffer = []

    def write(self, string):
        string = string.rstrip()
        if len(string) > 0:
            self.buffer.append(string)

    def __enter__(self):
        self.flush = sys.stdout.flush
        sys.stdout = self
        return self

    def __exit__(self, ext_type, exc_value, traceback):
        sys.stdout = sys.__stdout__


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def main():
    config = configparser.ConfigParser()
    config.read('config.ini')
    config_inputfile = config.get('GENERAL', 'inputfile')
    config_sample_method = config.get('CLASSIFIER', 'method')

    correctness = []
    f1 = []
    precision = []
    recall = []
    amc = []

    # set verbose
    config.set('GENERAL', 'verbose', 'on')

    print("start testing with method {0}.".format(config_sample_method))
    with PrintMagic() as pm:
        for i in range(1):
            s = mainprog(config)
            #eprint(pm.buffer)

            # parse lines
            l = pm.buffer
            correctness.append(s[0])
            f1.append(s[1])
            precision.append(s[2])
            recall.append(s[3])
            amc.append(s[4])

            pm.buffer = [] # clear buffer
            eprint('Iteration {0}'.format(i+1))

    # output result
    print('correctness:', sum(correctness) / float(len(correctness)))
    print('fl score:', sum(f1) / float(len(f1)))
    print('precision:', sum(precision) / float(len(precision)))
    print('recall:', sum(recall) / float(len(recall)))
    print('amount cost:', sum(amc) / float(len(amc)))

if __name__ == '__main__':
    main()
