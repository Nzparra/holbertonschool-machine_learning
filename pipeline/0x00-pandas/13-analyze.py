#!/usr/bin/env python3
"""
getting stats from dataframe
using knowledge DS4A
"""

import pandas as pd
from_file = __import__('2-from_file').from_file
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
stats = df.loc[:, df.columns != 'Timestamp'].describe()
print(stats)