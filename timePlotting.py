import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from matplotlib import dates as mpl_dates
import pandas as pd


# series = read_csv('Japan.csv', header=0, index_col=0, parse_dates=True, squeeze=True)
# print(series.head())

sp500  = pd.read_csv("SP500HistoricalData.csv") #Excel File
spDate = pd.to_datetime(sp500['Time'])
spDateList = sp500['Time'].tolist()
spChange = sp500['Percent'].tolist()

# plt.plot_date(spDate[:20],spChange[:20],linestyle='solid')
# plt.gcf().autofmt_xdate()
# date_format = mpl_dates.DateFormatter('%m-%d-%Y')
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.tight_layout()
# plt.show()

# # # To Plot a certain time period
# date1 = '1/3/00'
# date2 = '1/31/00'

# index1 = spDateList.index(date1)
# index2 = spDateList.index(date2)

# print(index1)
# print(index2)

# date 1 is starting date, date2 is ending date. The dates must be in sample date
# The dates go in the format of month-date-year
# Dates are passed in as string
def plotTimeSeries(date1, date2, dateList):
    index1 = dateList.index(date1)
    index2 = dateList.index(date2)
    if (date2 != dateList[-1]):
        plt.plot_date(spDate[index1:index2+1],spChange[index1:index2+1],linestyle='solid')
    else:
        plt.plot_date(spDate[index1:],spChange[index1:],linestyle='solid')

    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter('%m-%d-%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.show()

plotTimeSeries('1/3/00', '1/31/00', spDateList)