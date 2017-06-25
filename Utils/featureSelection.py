from sklearn.feature_selection import SelectPercentile, f_classif

# see sklearn SelectPercentile for more information.
methods = {'f_classif':f_classif}

def get_feature_selector(config):
    percentile = int(config.get('FEATURE SELECT', 'percentile'))
    method = config.get('FEATURE SELECT', 'method')
    if method == '':
        return None
    if method not in methods:
        print('Method not allowed! Returned.')
        return None
    selector = SelectPercentile(methods[method], percentile=percentile)
    return selector

def get_index(source, target):
    re = []
    sl = source.tolist()
    for i in target:
        try:
            re.append(sl.index(i))
        except ValueError:
            re.append('X')
    return re
