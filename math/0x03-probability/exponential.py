#!/usr/bin/env python3
""" Exponential """


class Exponential():
    """ Exponential """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Initialize """

        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1/sum(data)/len(data)
        else:
            lambtha = float(lambtha)
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha

    def pmf(self, x):
        """ PMF for a given number of “successes” """
        if x < 0:
            return 0
        return (self.lambtha * (Exponential.e ** (-self.lambtha * x))

    def cdf(self, x):
        """ CDF for a given number of “successes” """
        if k < 0:
            return 0
        return (1 - (Exponential.e ** (-self.lambtha * x)))
