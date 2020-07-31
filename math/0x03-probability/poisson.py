#!/usr/bin/env python3
""" Poisson """


class Poisson():
    """ Poisson """

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """ Initialize """

        if data is not None:
            if type(data) is not list:
                raise ValueError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)
        else:
            lambtha = float(lambtha)
            if lambtha < 1:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = lambtha

    def pmf(self, k):
        """ PMF for a given number of “successes” """
        if k < 0:
            return 0
        fact = 1
        for i in range(1, 1+k):
            fact = fact * i
        return ((Poisson.e ** (-self.lambtha)) * (self.lambtha ** k)) / fact

    def cdf(self, k):
        """ CDF for a given number of “successes” """
        k = int(k)
        if k < 0:
            return 0
        return sum([self.pmf(k) for k in range(0, 1 + k)])
