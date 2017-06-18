class BaseSampler:
    """
    This is an empty sampler.
    All sampler should inherit this class, and should override the fit_sample
    method.
    """

    def fit_sample(self, X, y):
        return X, y

