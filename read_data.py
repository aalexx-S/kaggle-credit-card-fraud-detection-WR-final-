def read_data(input_file):
    X = []
    y = []
    with open(input_file) as inf:
        # consume the first line which is tags
        inf.readline()
        for line in inf:
            sp = line.rstrip('\n').split(',')
            X.append([float(x) for x in sp[:-1]])
            y.append(int(sp[-1][1:-1]))
    return X, y
