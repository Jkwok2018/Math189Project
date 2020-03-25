
import numpy as np
import pandas as pd

sp500  = pd.read_csv("SP500HistoricalData.csv") #Excel File
jpn = pd.read_csv("Japan.csv")
hk = pd.read_csv("HSI.csv")
sh = pd.read_csv("Shanghai.csv")
sz = pd.read_csv("Shenzhen.csv")

sampleDate = pd.read_csv("SampleDate.csv")

spDate = sp500['Time'].tolist()
spChange = sp500['Percent'].tolist()

jpnDate = jpn['Time'].tolist()
jpnChange = jpn['Percent'].tolist()

hkDate = hk['Time'].tolist()
hkChange = hk['Percent'].tolist()

shDate = sh['Time'].tolist()
shChange = sh['Percent'].tolist()

szDate = sz['Date'].tolist()
szChange = sz['Percent'].tolist()

standardDate = sampleDate['Date'].tolist()
export = []

for i in range(len(standardDate)):
    new = []
    new.append(standardDate[i])
    if (standardDate[i] in spDate):
        new.append(spChange[spDate.index(standardDate[i])])
    else:
        new.append(0)
    if (standardDate[i] in jpnDate):
        new.append(jpnChange[jpnDate.index(standardDate[i])])
    else:
        new.append(0)
    if (standardDate[i] in hkDate):
        new.append(hkChange[hkDate.index(standardDate[i])])
    else:
        new.append(0)
    if (standardDate[i] in shDate):
        new.append(shChange[shDate.index(standardDate[i])])
    else:
        new.append(0)
    if (standardDate[i] in szDate):
        new.append(szChange[szDate.index(standardDate[i])])
    else:
        new.append(0)
    export.append(new)

df = pd.DataFrame(export, columns=['Date', "USA","Japan", "HK", "Shanghai", "Shenzhen"])    
df.to_excel(r'test2.xlsx', index = False, header = True)
