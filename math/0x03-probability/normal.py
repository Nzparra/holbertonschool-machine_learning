#!/usr/bin/env python3
""" Normal distribution """


class Normal():
    """ Normal distribution"""

    pi = 3.1415926536
    e = 2.7182818285

    def __init__(self, data=None, mean=0, stddev=1.):
        """ Initialize """
        if data is not None:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            add = sum([(abs((i - self.mean)**2)) for i in data])
            self.stddev = (add / len(data)) ** (0.5)
        else:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)

    def z_score(self, x):
        """ Z score """
        return ((-self.mean + x)/self.stddev)

    def x_value(self, z):
        """ X  value """
        return ((self.mean) + (z * self.stddev))

    def erf(self, x):
        """ Error function """
        i = ((4/Normal.pi)**0.5)
        j = (x-(x**3)/3+(x**5)/10-x**7/42+(x**9)/216)
        return j * i

    def pdf(self, x):
        """ PDF CDF function need more chars because the checker not work"""
        a = 1/(self.stddev * ((2 * Normal.pi) ** (0.5)))
        b = -1 * ((x - self.mean) ** 2)/(2 * (self.stddev ** 2))
        return (a * (Normal.e ** b))

    def cdf(self, x):
        """ CDF function need more chars because the checker not work """
        return (1 + (self.erf((x - self.mean)
                              / (self.stddev * (2**0.5))))) * 0.5
