# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 20:01:25 2021

@author: kesuiker
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd = pd.read_csv('covid.csv')

cali = pd[pd['Province_State'] == 'California']
cali_sum = cali.groupby(['Province_State']).sum()


cali_sum