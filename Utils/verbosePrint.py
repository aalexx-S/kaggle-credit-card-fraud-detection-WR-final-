class VerbosePrint:
    # make _verbose static
    _verbose = False

    @staticmethod
    def set_verbose():
        VerbosePrint._verbose = True

    @staticmethod
    def verbose_print(string):
        if VerbosePrint._verbose:
            print(string)
