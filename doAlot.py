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
    config_sample_method = config.get('SAMPLER', 'method')

    correctness = []
    f1 = []
    precision = []
    recall = []
    amc = []

    # set verbose
    config.set('GENERAL', 'verbose', 'on')

    print("start testing with method {0}.".format(config_sample_method))
    with PrintMagic() as pm:
        for i in range(10):
            mainprog(config)
            #eprint(pm.buffer)

            # parse lines
            l = pm.buffer
            correctness.append(float(l[10].split(':')[1]))
            f1.append(float(l[12].split(':')[1]))
            precision.append(float(l[13].split(':')[1]))
            recall.append(float(l[14].split(':')[1]))
            amc.append(float(l[15].split(':')[1]))

            pm.buffer = [] # clear buffer
            eprint('Iteration {0}'.format(i+1))

    # output result
    print('correctness: ' + np.mean(correctness))
    print('fl score: ' + np.mean(fl))
    print('precision: ' + np.mean(precision))
    print('recall: ' + np.mean(recall))
    print('amount cost: ' + np.mean(amc))

if __name__ == '__main__':
    main()
