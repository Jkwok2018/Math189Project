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

jpn = pd.read_csv('Japan.csv')
jpnDate = pd.to_datetime(jpn['Time'])
jpnDateList = jpn['Time'].tolist()
jpnChange = jpn['Percent'].tolist()

# plt.plot_date(spDate[:20],spChange[:20],linestyle='solid')
# plt.gcf().autofmt_xdate()
# date_format = mpl_dates.DateFormatter('%m-%d-%Y')
# plt.gca().xaxis.set_major_formatter(date_format)
# plt.tight_layout()
# plt.show()


# date 1 is starting date, date2 is ending date. The dates must be in sample date
# The dates go in the format of month-date-year
def plotTimeSeries(date1, date2, dateList):
    index1 = dateList.index(date1)
    index2 = dateList.index(date2)
    if (date2 != dateList[-1]):
        plt.plot_date(spDate[index1:index2+1],spChange[index1:index2+1],linestyle='solid', color = 'r')
    else:
        plt.plot_date(spDate[index1:],spChange[index1:],linestyle='solid')

    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter('%m-%d-%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.show()


# Plot two countries on top of one another
# Need to change jpnDate and spDate later 
def plotTimeSeries(date1, date2, dateList1, dateList2):
    index1 = dateList1.index(date1)
    index2 = dateList1.index(date2)

    L2index1 = dateList2.index(date1)
    L2index2 = dateList2.index(date2)
    if (date2 != dateList1[-1]):
        plt.plot_date(spDate[index1:index2+1],spChange[index1:index2+1],linestyle='solid', color = 'r')
        plt.plot_date(jpnDate[L2index1:L2index2+1],jpnChange[L2index1:L2index2+1],linestyle='solid', color = 'b')
    else:
        plt.plot_date(spDate[index1:],spChange[index1:],linestyle='solid', color = 'r')
        plt.plot_date(jpnDate[L2index1:],jpnChange[L2index1:],linestyle='solid', color = 'b')

    plt.gcf().autofmt_xdate()
    date_format = mpl_dates.DateFormatter('%m-%d-%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    plt.tight_layout()
    plt.show()   

plotTimeSeries('1/5/00', '1/31/00', spDateList, jpnDateList)