# -*- coding: utf-8 -*-
"""



Created on Sat Jan 30, 2021

@author: Kevin Suiker

co-author: Leonard the cat, for standing on the keyboard twice

Desscription:
    Covid death and Recovery for US and global
    
    
    
"""


import pandas as pd
import matplotlib.pyplot as plt

# URL for raw covid data
urlWorldDeath = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
urlUsDeath = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv'

urlWorldRecovered = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
 
# read in covid data to data frame
dfWorldDeath = pd.read_csv(urlWorldDeath)
dfUsDeath = pd.read_csv(urlUsDeath)

dfWorldRecovered = pd.read_csv(urlWorldRecovered)

dfUsRecovered = dfWorldRecovered[dfWorldRecovered['Country/Region'] == 'US']
                                 

#sum the numbers to create totals for global and US, we only want the numbers and not county/country etc.
worldDeathSum = dfWorldDeath.iloc[:, 4:380].sum()
worldDeathSum.index = pd.to_datetime(worldDeathSum.index)

worldRecoveredSum = dfWorldRecovered.iloc[:, 4:380].sum()
worldRecoveredSum.index = pd.to_datetime(worldRecoveredSum.index)

usRecoveredSum = dfUsRecovered.iloc[:, 4:380].sum()
usRecoveredSum.index = pd.to_datetime(usRecoveredSum.index)

usDeathSum = dfUsDeath.iloc[:, 12:388].sum()
usDeathSum.index = pd.to_datetime(usDeathSum.index)

# Create subplots
fig,axs = plt.subplots(2)

fig.suptitle("Covid-19 Deaths for US and Global")

#plot and label the goods
axs[0].plot(usDeathSum)
axs[0].set_title("US")
axs[0].set_xlabel("Date")
axs[0].set_ylabel("Deaths")


axs[1].plot(worldDeathSum)
axs[1].set_title("Global")
axs[1].set_xlabel("Date")
axs[1].set_ylabel("Deaths")

# seperate subplot for recoveries 
fig2,axs2 = plt.subplots(2)

fig2.suptitle("Covid-19 Recoveries for US and Global")

axs2[0].plot(usRecoveredSum)
axs2[0].set_title("US")
axs2[0].set_xlabel("Date")
axs2[0].set_ylabel("Recovered")


axs2[1].plot(worldRecoveredSum)
axs2[1].set_title("Global")
axs2[1].set_xlabel("Date")
axs2[1].set_ylabel("Recovered")