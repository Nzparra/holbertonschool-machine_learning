#!/usr/bin/env python3
""" calculates the weighted moving average of a data set """


def moving_average(data, beta):
    """
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
    """
    moving = []
    V = 0
    for i in range(1, len(data) + 1):
        fix = ((beta * V) + ((1 - beta) * data[i - 1])) / (1 - (beta ** i))
        moving.append(fix)
        V = ((beta * V) + ((1 - beta) * data[i - 1]))
    return moving
