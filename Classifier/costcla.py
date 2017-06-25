import numpy as np
# eliminate sklearn's deprecation warning from costcla.
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from math import sqrt
from ast import literal_eval
from costcla.models import CostSensitiveLogisticRegression
from costcla.models import CostSensitiveRandomForestClassifier

from .baseClassifier import BaseClassifier
from .randomForest import RandomForest
from .logisticRegression import LogisticRegression

class Costcla(BaseClassifier):
    def __init__(self, config):
        self.config = self._Parameters(config)
        if self.config.method == 'logistic regression':
            config = LogisticRegression._Parameters(config)
            self.clf = CostSensitiveLogisticRegression(**config.kwargs)
        elif self.config.method == 'random forest':
            config = RandomForest._Parameters(config)
            del config.kwargs['max_depth']
            config.kwargs['combination'] = 'stacking'
            config.kwargs['max_features'] = 'auto'
            config.kwargs['n_jobs'] = -1
            self.clf = CostSensitiveRandomForestClassifier(**config.kwargs)

    def fit(self, X, y):
        cost_matrix = self.compute_cost_matrix(X, y)
        self.clf.fit(X, y, cost_matrix)

    def predict(self, X):
        return self.clf.predict(X)

    def setArgs(self, train_X_amount=None):
        self.train_X_amount = train_X_amount

    # cost_mat[C_FP,C_FN,C_TP,C_TN]
    def compute_cost_matrix(self, X, y):
        const = self.config.const
        FP_methods={'sqrt':sqrt, '':lambda x: const
                    ,'ratio':lambda x: x*self.config.FP_ratio}
        FN_methods={'sqrt':sqrt, '':lambda x: const
                    ,'ratio':lambda x: x*self.config.FN_ratio}
        re = []
        mean_amount = np.mean(self.train_X_amount)
        for i, j in zip(self.train_X_amount, y):
            C_FP = FP_methods[self.config.cost_FP_method](i[0])
            C_FN = FN_methods[self.config.cost_FN_method](i[0])
            cm = [max(C_FP, const), max(C_FN, const), const, 0]
            re.append(cm)
        return np.array(re)

    class _Parameters:
        def __init__(self, config):
            allowed = literal_eval(config.get('COSTCLA', 'allowed'))
            self.method = config.get('COSTCLA', 'method')
            if self.method not in allowed:
                raise ValueError('Costcla method not allowed by'
                    + ' config.COSTCLA.allowed.')
            allowed = literal_eval(config.get('COSTCLA', 'cost_allowed'))
            # cost
            self.cost_FP_method = config.get('COSTCLA', 'FP_method')
            self.cost_FN_method = config.get('COSTCLA', 'FN_method')
            if (self.cost_FP_method not in allowed
                    or self.cost_FN_method not in allowed):
                raise ValueError('Costcla cost matrix method not allowed by'
                    + ' config.COSTCLA.cost_allowed.')
            self.FP_ratio = float(config.get('COSTCLA', 'FP_ratio'))
            self.FN_ratio = float(config.get('COSTCLA', 'FN_ratio'))
            self.const = int(config.get('COSTCLA', 'const'))

