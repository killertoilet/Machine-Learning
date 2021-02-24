# -*- coding: utf-8 -*-
"""


Created on Sat Jan 30, 2021

@author: Kevin Suiker


Desscription:
    Covid confirmed cases for 5 states.
        -Totals
        -per 100k people
    
    
    
"""
import pandas as pd
import matplotlib.pyplot as plt


usConfirmedUrl = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"

dfUsConfirmed = pd.read_csv(usConfirmedUrl)

cali = dfUsConfirmed[dfUsConfirmed['Province_State'] == 'California']
caliSum = cali.iloc[:,11:387].sum()
caliSum.index = pd.to_datetime(caliSum.index)

florida = dfUsConfirmed[dfUsConfirmed['Province_State'] == 'Florida']
floridaSum = florida.iloc[:,11:387].sum()
floridaSum.index = pd.to_datetime(floridaSum.index)

texas = dfUsConfirmed[dfUsConfirmed['Province_State'] == 'Texas']
texasSum = texas.iloc[:,11:387].sum()
texasSum.index = pd.to_datetime(texasSum.index)

newYork = dfUsConfirmed[dfUsConfirmed['Province_State'] == 'New York']
newYorkSum = newYork.iloc[:,11:387].sum()
newYorkSum.index = pd.to_datetime(newYorkSum.index)

wyoming = dfUsConfirmed[dfUsConfirmed['Province_State'] == 'Wyoming']
wyomingSum = wyoming.iloc[:,11:387].sum()
wyomingSum.index = pd.to_datetime(wyomingSum.index)

df = dfUsConfirmed[dfUsConfirmed['Province_State'].isin(['California','Florida','Texas','New York','Wyoming'])]

# pulled from census data
caliPop = 37253956
floridaPop = 18801310
texasPop = 25145561
newYorkPop = 19378102
wyomingPop = 563626

# calculating per 100k pop
caliSumPer100 = caliSum / caliPop * 100000
floridaSumPer100 = floridaSum / floridaPop * 100000
texasSumPer100 = texasSum / texasPop * 100000
newYorkSumPer100 = newYorkSum / newYorkPop * 100000
wyomingSumPer100 = wyomingSum / wyomingPop * 100000

caliSumPer100.index = pd.to_datetime(caliSum.index)
floridaSumPer100.index = pd.to_datetime(caliSum.index)
texasSumPer100.index = pd.to_datetime(caliSum.index)
newYorkSumPer100.index = pd.to_datetime(caliSum.index)
wyomingSumPer100.index = pd.to_datetime(caliSum.index)

#one subplot for totals, another for per 100k population
fig,axs = plt.subplots(6)

fig.suptitle("Case Totals for 5 US States")

axs[0].plot(caliSum)
axs[0].set_title('California')

axs[1].plot(floridaSum)
axs[1].set_title("Florida")

axs[2].plot(texasSum)
axs[2].set_title("Texas")

axs[3].plot(newYorkSum)
axs[3].set_title("New York")

axs[4].plot(wyomingSum)
axs[4].set_title("Wyoming")

axs[5].plot(caliSum, label='Cali')
axs[5].plot(floridaSum, label='Florida')
axs[5].plot(texasSum, label='Texas')
axs[5].plot(newYorkSum, label = 'New York')
axs[5].plot(wyomingSum, label='Wyoming')
axs[5].legend()

fig2,axs2 = plt.subplots(6)

fig2.suptitle("Cases Per 100k Population")

axs2[0].plot(caliSumPer100)
axs2[0].set_title('California')

axs2[1].plot(floridaSumPer100)
axs2[1].set_title('Florida')

axs2[2].plot(texasSumPer100)
axs2[2].set_title("Texas")

axs2[3].plot(newYorkSumPer100)
axs2[3].set_title("New York")

axs2[4].plot(wyomingSumPer100)
axs2[4].set_title("Wyoming")

axs2[5].plot(caliSumPer100, label='Cali')
axs2[5].plot(floridaSumPer100, label='Florida')
axs2[5].plot(texasSumPer100, label='Texas')
axs2[5].plot(newYorkSumPer100, label = 'New York')
axs2[5].plot(wyomingSumPer100, label='Wyoming')
axs2[5].legend()

